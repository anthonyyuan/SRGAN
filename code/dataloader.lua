local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

local M = {}
local DataLoader = torch.class('resnet.DataLoader', M)

function DataLoader.create(opt)
    print('loading data...')
    local loaders = {}
    for i, split in ipairs{'train', 'val'} do
        local dataset = require('datasets/' .. opt.dataset)(opt,split)
        print('\tInitializing data loader for ' .. split .. ' set...')
        loaders[i] = M.DataLoader(dataset, opt, split)
    end
    return table.unpack(loaders)
end

function DataLoader:__init(dataset, opt, split)
    local manualSeed = opt.manualSeed
    local function init()
        require('datasets/' .. opt.dataset)
    end
    local function main(idx)
        if manualSeed ~= 0 then
            torch.manualSeed(manualSeed + idx)
        end
        torch.setnumthreads(1)
        _G.dataset = dataset
        _G.augment = dataset:augment()
        return dataset:size()
    end

    local threads, sizes = Threads(opt.nThreads,init,main)
    self.threads = threads
    self.__size = sizes[1][1]
    self.batchSize = opt.batchSize
    self.split = split
    self.opt = opt
end

function DataLoader:size()
    return math.ceil(self.__size / self.batchSize)
end

function DataLoader:run()
    local threads = self.threads
    threads:synchronize()
    local size, batchSize = self.__size, self.batchSize
    local perm = torch.randperm(size)
    perm = torch.randperm(size)

    local idx, sample = 1, nil
    local function enqueue()
        if self.split == 'train' then
            while idx <= size and threads:acceptsjob() do
                local indices = perm:narrow(1, idx, math.min(batchSize, size - idx + 1))
                local patchSize,scale = self.opt.patchSize,self.opt.scale
                threads:addjob(
                    function(indices)
                        local batchSize = indices:size(1)
                        local inpSize,tarSize = patchSize/scale,patchSize
                        local input_batch = torch.Tensor(batchSize,3,inpSize,inpSize):zero()
                        local target_batch = torch.Tensor(batchSize,3,tarSize,tarSize):zero()

                        for i,idx in ipairs(indices:totable()) do
                            local idx_ = idx
                            ::redo::
                            local sample = _G.dataset:get(idx_)
                            if not sample then 
                                idx_ = idx_ + 1
                                goto redo
                            end

                            sample = _G.augment(sample)

                            input_batch[i]:copy(sample.input)
                            target_batch[i]:copy(sample.target)
                        end
                        collectgarbage()
                        return {
                            input = input_batch,
                            target = target_batch,
                        }
                    end,
                    function (_sample_)
                        sample = _sample_
                        return sample
                    end,
                    indices
                )
                idx = idx + batchSize
            end
        elseif self.split == 'val' then
            while idx <= size and threads:acceptsjob() do
                threads:addjob(
                    function(idx)
                        local sample = _G.dataset:get(idx)
                        return {
                            input = sample.input,
                            target = sample.target
                        }
                    end,
                    function (_sample_)
                        sample = _sample_
                        return sample
                    end,
                    idx
                )        
                idx = idx + 1
            end
        end 
    end

    local n = 0
    local function loop()
        enqueue()
        if not threads:hasjob() then
            return nil
        end
        threads:dojob()
        if threads:haserror() then
            threads:synchronize()
        end
        enqueue()
        n = n + 1
        return n, sample
    end

    return loop
end

return M.DataLoader
