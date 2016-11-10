require 'nn'
require 'cunn'

local function createModel(opt)
    local conv = nn.SpatialConvolution
    local relu = nn.ReLU
    local bnorm = nn.SpatialBatchNormalization

    local function basicblock(nFeat, stride, preActivation)
        local s = nn.Sequential()
        if not preActivation then
            s:add(conv(nFeat,nFeat,3,3,stride,stride,1,1))
            s:add(bnorm(nFeat))
            s:add(relu(true))
            s:add(conv(nFeat,nFeat,3,3,1,1,1,1))
            s:add(bnorm(nFeat))
        else
            s:add(bnorm(nFeat))
            s:add(relu(true))
            s:add(conv(nFeat,nFeat,3,3,stride,stride,1,1))
            s:add(bnorm(nFeat))
            s:add(relu(true))
            s:add(conv(nFeat,nFeat,3,3,1,1,1,1))
        end

        return nn.Sequential()
            :add(nn.ConcatTable()
                :add(s)
                :add(nn.Identity()))
            :add(nn.CAddTable(true))
    end

    local function bottleneck(nFeat, stride, preActivation)
        local s = nn.Sequential()
        if not preActivation then
            s:add(conv(nFeat,nFeat,1,1))
            s:add(bnorm(nFeat))
            s:add(relu(true))
            s:add(conv(nFeat,nFeat,3,3,stride,stride,1,1))
            s:add(bnorm(nFeat))
            s:add(relu(true))
            s:add(conv(nFeat,nFeat,1,1))
            s:add(bnorm(nFeat))
        else
            s:add(bnorm(nFeat))
            s:add(relu(true))
            s:add(conv(nFeat,nFeat,1,1))
            s:add(bnorm(nFeat))
            s:add(relu(true))
            s:add(conv(nFeat,nFeat,3,3,stride,stride,1,1))
            s:add(bnorm(nFeat))
            s:add(relu(true))
            s:add(conv(nFeat,nFeat,1,1))
        end

        return nn.Sequential()
            :add(nn.ConcatTable()
                :add(s)
                :add(nn.Identity()))
            :add(nn.CAddTable(true))
    end
    
    local filt_deconv = opt.filt_deconv
    local filt_recon = opt.filt_recon
    local pad_deconv = (filt_deconv-1)/2
    local pad_recon = (filt_recon-1)/2
    local preActivation = false
    if opt.netType == 'preResNet' then preActivation = true end
    local conv_block = basicblock
    if opt.bottleneck then conv_block = bottleneck end

    local model = nn.Sequential()
        :add(conv(opt.nChannel,opt.nFeat, 3,3, 1,1, 1,1))
        :add(relu(true))
    for i=1,opt.nResBlock do
        model:add(conv_block(opt.nFeat, 1, preActivation))
    end
    if opt.netType == 'preResNet' then
        model:remove(2) -- remove relu (duplicated)
        model:remove(2) -- remove bnorm (not appeared in the paper)
        model:add(bnorm(opt.nFeat))
        model:add(relu(true))
    end
    if opt.upsample == 'full' then
        model:add(nn.SpatialFullConvolution(opt.nFeat,opt.nFeat, filt_deconv,filt_deconv, 2,2, pad_deconv,pad_deconv, 1,1))
        model:add(relu(true))
        model:add(nn.SpatialFullConvolution(opt.nFeat,opt.nFeat, filt_deconv,filt_deconv, 2,2, pad_deconv,pad_deconv, 1,1))
        model:add(relu(true))
    -- bilinear upsampling seems not supporting GPU version yet. (nov. 7, 2016)
    elseif opt.upsample == 'bilinear' then
        model:add(nn.SpatialUpSamplingBilinear({owidth=opt.patchSize,oheight=opt.patchSize}))
        model:add(conv(opt.nFeat,opt.nFeat,filt_deconv,filt_deconv,1,1,pad_deconv,pad_deconv))
        model:add(relu(true))
        model:add(nn.SpatialUpSamplingBilinear({owidth=opt.patchSize,oheight=opt.patchSize}))
        model:add(conv(opt.nFeat,opt.nFeat,filt_deconv,filt_deconv,1,1,pad_deconv,pad_deconv))
        model:add(relu(true))
    end


    model:add(conv(opt.nFeat,opt.nChannel, filt_recon,filt_recon, 1,1, pad_recon,pad_recon))

    --model:insert(nn.Copy(opt.defaultType,opt.operateType),1)
    if opt.normalize then
        local mean = torch.Tensor({0.485,0.456,0.406})
        local subMean = nn.SpatialConvolution(3,3,1,1)
        subMean.weight = torch.eye(3,3):view(3,3,1,1)
        subMean.bias = torch.Tensor(mean):mul(-1)
        local addMean = nn.SpatialConvolution(3,3,1,1)
        addMean.weight = torch.eye(3,3):view(3,3,1,1)
        addMean.bias = torch.Tensor(mean)
        local std = torch.Tensor({0.229,0.224,0.225})
        local divStd = nn.SpatialConvolution(3,3,1,1):noBias()
        divStd.weight = torch.Tensor({{1/std[1],0,0},{0,1/std[2],0},{0,0,1/std[3]}})
        local mulStd = nn.SpatialConvolution(3,3,1,1):noBias()
        mulStd.weight = torch.Tensor({{std[1],0,0},{0,std[2],0},{0,0,std[3]}})

        model:insert(subMean,1)
        model:insert(divStd,2)
        model:add(mulStd)
        model:add(addMean)
    end

    return model
end

return createModel
