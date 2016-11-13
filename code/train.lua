local image = require 'image'
local optim = require 'optim'

local M = {}
local Trainer = torch.class('resnet.Trainer', M)

function Trainer:__init(model, criterion, opt, optimState)
    self.model = model
    self.criterion = criterion
    self.optimState = optimState or opt.optimState
    self.opt = opt
    self.errG, self.errD = 0,0
    self.params, self.gradParams = {},{}
    self.params.G, self.gradParams.G = model:getParameters()
    self.feval = {}
    if opt.adv > 0 then
        local iAdvLoss = self.criterion.iAdvLoss
        self.advLoss = self.criterion.criterions[iAdvLoss] -- nn.AdversarialLoss (see 'loss/init.lua' and 'loss/AdversarialLoss.lua')
        self.params.D, self.gradParams.D = self.advLoss.discriminator:getParameters()
        self.feval.D = function() return self.errD, self.gradParams.D end
    end
    self.feval.G = function() return self.errG, self.gradParams.G end
end

function Trainer:train(epoch, dataloader)

    local size = dataloader:size()
    local timer = torch.Timer()
    local dataTimer = torch.Timer()
    local trainTime, dataTime = 0,0
    local iter, avgLossG, avgLossD = 0,0,0

    self.model:training()
    for n, sample in dataloader:run() do
        dataTime = dataTime + dataTimer:time().real

        if self.opt.nGPU > 0 then
            self:copyInputs(sample) -- Copy input and target to the GPU
        end

        -- Train generator network
        self.model:zeroGradParameters()

        self.model:forward(self.input):float()
        self.errG = self.criterion(self.model.output, self.target)
        self.model:backward(self.input, self.criterion.gradInput)

        self.optimState.method(self.feval.G, self.params.G, self.optimState)

        -- Train adversarial network
        if self.opt.adv > 0 then
            self.advLoss.discriminator:zeroGradParameters()
            local errD_fake = self.advLoss:accGradParameters(self.model.output,'fake')
            local errD_real = self.advLoss:accGradParameters(self.target,'real')
  
            self.errD = (errD_fake + errD_real) / 2

            self.optimState.method(self.feval.D, self.params.D, self.optimState)
        end

        avgLossG = avgLossG + self.errG
        avgLoggD = avgLossD + self.errD

        trainTime = trainTime + timer:time().real
        timer:reset()
        dataTimer:reset()

        iter = iter + 1
        if n % self.opt.printEvery == 0 then
            print(('[%d/%d] Time: %.3f   Data: %.3f  avgLossG: %.6f  avgLossD: %.6f')
                :format(n,size,trainTime,dataTime,avgLossG/iter,avgLossD/iter))
            if n % self.opt.testEvery ~= 0 then
                avgLossG, avgLossD = 0,0
                trainTime, dataTime = 0,0
                iter = 0
            end
        end

        if n % self.opt.testEvery == 0 then break end
    end
    
    return avgLossG/iter, avgLossD/iter
end

function Trainer:test(epoch, dataloader)

    local timer = torch.Timer()
    local dataTimer = torch.Timer()

    local iter,avgPSNR = 0,0

    self.model:evaluate()
    for n, sample in dataloader:run() do
        local dataTime = dataTimer:time().real

        if self.opt.nGPU > 0 then
            self:copyInputs(sample)
        end

        local output, target = self.input, self.target
        output = output:view(1,table.unpack(output:size():totable()))
        target = target:view(1,table.unpack(target:size():totable()))

        local model = self.model
        if torch.type(self.model)=='nn.DataParallelTable' then model = model:get(1) end

        for i=1,#model do
            local block = model:get(i):clone('weight','bias')
            output = block:forward(output)
        end
        collectgarbage()

        local psnr = self:calcPSNR(output,target)
        avgPSNR = avgPSNR + psnr
        iter = iter + 1

        image.save(paths.concat(self.opt.save,'result',n .. '.jpg'), output:float():squeeze())
        --print( (' Test: [%d][%d/%d]   Time: %.3f   Data: %.3f   PSNR: %.2f'):format(
            --epoch, n, size, timer:time().real, dataTime, psnr) )

        --timer:reset()
        --dataTimer:reset()
    end

    self.model:training()

    print(('[epoch %d (iter/epoch: %d)] Average PSNR: %.2f, Test time: %.2f\n'):format(epoch, self.opt.testEvery, avgPSNR / iter, timer:time().real))

    return avgPSNR / iter
end

function Trainer:copyInputs(sample)
   -- Copies the input to a CUDA tensor, if using 1 GPU, or to pinned memory, if using DataParallelTable. 
   self.input = self.input or (self.opt.nGPU == 1 and torch.CudaTensor() or cutorch.createCudaHostTensor())
   --self.target = self.target or (self.opt.nGPU == 1 and torch.CudaTensor() or cutorch.createCudaHostTensor())
   self.target = self.target or torch.CudaTensor()
   self.input:resize(sample.input:size()):copy(sample.input)
   self.target:resize(sample.target:size()):copy(sample.target)
end

function Trainer:calcPSNR(output,target)
    local sc = self.opt.scale

    output = image.rgb2y(output:float():squeeze()):squeeze()
    target = image.rgb2y(target:float():squeeze()):squeeze()
    local h,w = table.unpack(output:size():totable())

    local diff = output[{{sc+1,h-sc},{sc+1,w-sc}}] - target[{{sc+1,h-sc},{sc+1,w-sc}}]

    local mse = diff:pow(2):mean()
    local psnr = -10*math.log10(mse)

    return psnr
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
