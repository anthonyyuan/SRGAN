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
			:add(nn.SpatialBatchNormalization(nOuptutPlane))
		return block
	end

    local negval = opt.negval
    local filtsize_1 = 3
    local filtsize_2 = opt.fitsizeD
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

    if opt.nGPU > 0 then
        if opt.backend == 'cudnn' then
            self.discriminator = cudnn.convert(self.discriminator,cudnn)
        end
        self.discriminator:cuda()
        self.crit:cuda()
    end
end

function AdversarialLoss:updateOutput(input,target,mode)
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
end

function AdversarialLoss:updateGradInput(input,target)
    self.dl_do = self.crit:backward(self.adv_output,self.adv_target)
    self.gradInput = self.discriminator:updateGradInput(input,self.dl_do)
    
    return self.gradInput
end

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