require 'nn'
require 'cunn'

local function createModel(opt)
    local conv = nn.SpatialConvolution
    local relu = nn.ReLU
    local bnorm = nn.SpatialBatchNormalization

    local filt_recon = opt.filt_recon
    local pad_recon = (filt_recon-1)/2

    local model = nn.Sequential()
    local main = nn.Sequential()
    main:add(conv(opt.nChannel,opt.nFeat, filt_recon,filt_recon, 1,1, pad_recon, pad_recon))
    main:add(relu(true))
    for i = 1,opt.nLayer do
        main:add(conv(opt.nFeat,opt.nFeat, 3,3, 1,1, 1,1))
        main:add(bnorm(opt.nFeat))
        main:add(relu(true))
    end
    main:add(conv(opt.nFeat,opt.nChannel, filt_recon,filt_recon, 1,1, pad_recon,pad_recon))
    local concat = nn.ConcatTable()
    concat:add(main):add(nn.Identity())
    model:add(concat)
    model:add(nn.CAddTable())

    return model
end

return createModel