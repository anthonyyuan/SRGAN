require 'nn'
require 'cunn'
require 'cudnn'

local AdversarialLoss, parent = torch.class('nn.AdversarialLoss','nn.Criterion')

function AdversarialLoss:__init(opt)
	local function conv_block(nInputPlane, nOutputPlane, filtsize, str)
		local pad = math.floor((filtsize-1)/2)
        local negval = opt.negval
		local block = nn.Sequential()
			:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, filtsize, filtsize, str,str, pad,pad))
			:add(nn.LeakyReLU(negval, true))
			:add(nn.SpatialBatchNormalization(nOutputPlane))
		return block
	end

    local negval = opt.negval
    local filtsize_1 = 3
    local filtsize_2 = opt.filtsizeD
    local pad = (filtsize_1 - 1) / 2
    local ks = opt.patchSize / (2^4)

    local discriminator = nn.Sequential()
        :add(nn.SpatialConvolution(3,64,filtsize_1,filtsize_1,1,1,pad,pad))
        :add(nn.LeakyReLU(negval,true))
        :add(conv_block(64,64,filtsize_2,2))
        :add(conv_block(64,128,filtsize_1,1))
        :add(conv_block(128,128,filtsize_2,2))
        :add(conv_block(128,256,filtsize_1,1))
        :add(conv_block(256,256,filtsize_2,2))
        :add(conv_block(256,512,filtsize_1,1))
        :add(conv_block(512,512,filtsize_2,2))
        :add(nn.SpatialConvolution(512,1024,ks,ks)) -- dense
        :add(nn.LeakyReLU(negval,true))
        :add(nn.SpatialConvolution(1024,1,1,1)) -- dense
        :add(nn.Sigmoid())

    self.discriminator = discriminator
    self.crit = nn.BCECriterion()

    self.err = 0
    self.params, self.gradParams = self.discriminator:getParameters()
    self.feval = function() return self.err, self.gradParams end
    self.optimState = opt.optimState_D

    if opt.nGPU > 0 then
        if opt.backend == 'cudnn' then
            self.discriminator = cudnn.convert(self.discriminator,cudnn)
        end
        self.discriminator:cuda()
        self.crit:cuda()
    end
end

function AdversarialLoss:updateOutput(input,target,mode)

    self.d_output_fake = self.discriminator:forward(input):clone()
    self.d_target_real = self.d_output_fake.new():resizeAs(self.d_output_fake):fill(1)

    self.output = self.crit:forward(self.d_output_fake,self.d_target_real)

    return self.output
--[[
    self.adv_output = self.discriminator:forward(input):clone()
    local mode = mode or 'real'
    if mode == 'fake' then
        self.adv_target = self.adv_output.new():resizeAs(self.adv_output):fill(0)
    elseif mode == 'real' then
        self.adv_target = self.adv_output.new():resizeAs(self.adv_output):fill(1)
    else
        error('Invalid adversarial loss mode')
    end
    self.output = self.crit:forward(self.adv_output,self.adv_target)
    return self.output
--]]
end

function AdversarialLoss:updateGradInput(input,target)
    self.gradOutput = self.crit:backward(self.d_output_fake,self.d_target_real)
    self.gradInput = self.discriminator:updateGradInput(input,self.gradOutput) -- return value

    -- discriminator train
    self.discriminator:zeroGradParameters()
        -- fake
    self.d_target_fake = self.d_output_fake.new():resizeAs(self.d_output_fake):fill(0)
    self.err_fake = self.crit:forward(self.d_output_fake,self.d_target_fake)
    local gradOutput_fake = self.crit:backward(self.d_output_fake,self.d_target_fake)
    self.discriminator:backward(input,gradOutput_fake)
        -- real
    self.d_output_real = self.discriminator:forward(target):clone()
    self.err_real = self.crit:forward(self.d_output_real,self.d_target_real)
    local gradOutput_real = self.crit:backward(self.d_output_real,self.d_target_real)
    self.discriminator:backward(target,gradOutput_real) 

    print('optim D')
    self.optimState.method(self.feval, self.params, self.optimState)

    return self.gradInput

--[[
    self.dl_do = self.crit:backward(self.adv_output,self.adv_target)
    self.gradInput = self.discriminator:updateGradInput(input,self.dl_do)
    return self.gradInput
--]]
end
--[[
function AdversarialLoss:accGradParameters(input,mode)
    local errD
    if mode == 'fake' then
        errD = self.output
        self.adv_target:fill(0)
    elseif mode == 'real' then
        errD = self:updateOutput(input,'real')
    end

    self:updateGradInput(input)
    self.discriminator:accGradParameters(input,self.dl_do)

    return errD
end
--]]