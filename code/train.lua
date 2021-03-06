local image = require 'image'
local optim = require 'optim'
local util = require 'utils'()

local M = {}
local Trainer = torch.class('sr.Trainer', M)

function Trainer:__init(model, criterion, opt)
    self.model = model
    self.criterion = criterion
    self.opt = opt
    self.optimState = opt.optimState_G

    self.err = 0
    self.params, self.gradParams = model:getParameters()
    self.feval = function() return self.err, self.gradParams end

    if opt.adv > 0 then
        local iAdvLoss = self.criterion.iAdvLoss
        self.advLoss = self.criterion.criterions[iAdvLoss] -- nn.AdversarialLoss (see 'loss/init.lua' and 'loss/AdversarialLoss.lua')
    end
    if opt.tv > 0 then
        local iTVLoss = opt.adv > 0 and self.criterion.iAdvLoss + 1 or self.criterion.iAdvLoss
        self.TVLoss = self.criterion.criterions[iTVLoss]
    end
end

function Trainer:train(epoch, dataloader)

    local size = dataloader:size()
    local timer = torch.Timer()
    local dataTimer = torch.Timer()
    local trainTime, dataTime = 0,0
    local iter, errG, errG_adv, errG_tv = 0,0,0,0
    local errD_fake, errD_real = 0,0

    self.model:training()
    for n, sample in dataloader:run() do
        dataTime = dataTime + dataTimer:time().real

        if self.opt.nGPU > 0 then
            self:copyInputs(sample,'train') -- Copy input and target to the GPU
        end

        self.model:zeroGradParameters()

        self.model:forward(self.input)
        self.err = self.criterion(self.model.output, self.target)
        self.model:backward(self.input, self.criterion.gradInput)

        if self.opt.clip > 0 then
            self.gradParams:clamp(-self.opt.clip/self.opt.lr,self.opt.clip/self.opt.lr)
        end
        self.optimState.method(self.feval, self.params, self.optimState)

        errG = errG + self.err
        if self.opt.adv > 0 then
            errG_adv = errG_adv + self.advLoss.output
            errD_fake = errD_fake + self.advLoss.err_fake
            errD_real = errD_real + self.advLoss.err_real
        end
        if self.opt.tv > 0 then
            errG_tv = errG_tv + self.TVLoss.output
        end

        trainTime = trainTime + timer:time().real
        timer:reset()
        dataTimer:reset()

        iter = iter + 1
        if n % self.opt.printEvery == 0 then
            print(('[%d/%d] Time: %.3f (data: %.3f),    errG: %.6f (adv: %.6f)(tv: %.6f),    errD_fake: %.6f  errD_real: %.6f')
                :format(n,self.opt.testEvery,trainTime,dataTime,
                    errG/iter, self.opt.adv*errG_adv/iter, self.opt.tv*errG_tv/iter,
                    errD_fake/iter, errD_real/iter))
            if n % self.opt.testEvery ~= 0 then
                errG, errG_adv, errG_tv = 0,0,0
                errD_fake, errD_real = 0,0
                trainTime, dataTime = 0,0
                iter = 0
            end
        end

        if n % self.opt.testEvery == 0 then break end
    end
    
    return errG/iter
end

function Trainer:test(epoch, dataloader)
    local timer = torch.Timer()
    local iter,avgPSNR = 0,0
    if self.opt.netType=='VDSR' then avgPSNR = torch.zeros(3) end

    self.model:evaluate()
    for n, sample in dataloader:run() do
        local function test_img(scale)
            if self.opt.nGPU > 0 then
                if self.opt.netType=='VDSR' then
                    self:copyInputs({input = sample.input[scale-1],
                                    target = sample.target[scale-1]},'test')
                else
                    self:copyInputs(sample,'test')
                end
            end

            local output, target = self.input, self.target
            output = output:view(1,table.unpack(output:size():totable()))
            target = target:view(1,table.unpack(target:size():totable()))
            if self.opt.nChannel==1 then
                output = output:view(1,table.unpack(output:size():totable()))
                target = target:view(1,table.unpack(target:size():totable()))
            end

            local model = self.model
            if torch.type(self.model)=='nn.DataParallelTable' then model = model:get(1) end

            output = output:cuda()
            for i=1,#model do
                local block = model:get(i):clone('weight','bias')
                output = block:forward(output)
            end
            collectgarbage()
            output = output:squeeze(1)
            target = target:squeeze(1)

            local psnr = util:calcPSNR(output,target,scale)

            if self.opt.netType=='VDSR' then
                avgPSNR[scale-1] = avgPSNR[scale-1] + psnr
                image.save(paths.concat(self.opt.save,'result',n ..'_x'..scale..'.jpg'), output:float():squeeze():div(255))
            else
                avgPSNR = avgPSNR + psnr
                image.save(paths.concat(self.opt.save,'result',n .. '.jpg'), output:float():squeeze():div(255))
            end
        end

        if self.opt.netType=='VDSR' then
            for scale=2,4 do
                test_img(scale)
            end
        else
            test_img(self.opt.scale)
        end
        iter = iter + 1
    end

    self.model:training()

    if self.opt.netType=='VDSR' then
        print(('[epoch %d (iter/epoch: %d)] Average PSNR: %.2f / %.2f / %.2f,  Test time: %.2f\n')
            :format(epoch, self.opt.testEvery, avgPSNR[1]/iter, avgPSNR[2]/iter, avgPSNR[3]/iter, timer:time().real))
    else
        print(('[epoch %d (iter/epoch: %d)] Average PSNR: %.2f,  Test time: %.2f\n')
            :format(epoch, self.opt.testEvery, avgPSNR / iter, timer:time().real))
    end

    return avgPSNR / iter
end

function Trainer:copyInputs(sample,mode)
    -- Copies the input to a CUDA tensor, if using 1 GPU, or to pinned memory, if using DataParallelTable. 
    if mode=='train' then
        self.input = self.input or (self.opt.nGPU == 1 and torch.CudaTensor() or cutorch.createCudaHostTensor())
    elseif mode=='test' then
        self.input = self.input or torch.CudaTensor()
    end
    --self.target = self.target or (self.opt.nGPU == 1 and torch.CudaTensor() or cutorch.createCudaHostTensor())
    self.target = self.target or torch.CudaTensor()
    self.input:resize(sample.input:size()):copy(sample.input)
    self.target:resize(sample.target:size()):copy(sample.target)
end

--[[
function Trainer:learningRate(epoch)
   -- Training schedule
   local decay = 0
   if self.opt.dataset == 'imagenet' then
      decay = math.floor((epoch - 1) / 30)
   elseif self.opt.dataset == 'cifar10' then
      decay = epoch >= 122 and 2 or epoch >= 81 and 1 or 0
   elseif self.opt.dataset == 'cifar100' then
      decay = epoch >= 122 and 2 or epoch >= 81 and 1 or 0
   end
   return self.opt.LR * math.pow(0.1, decay)
end
--]]

return M.Trainer
