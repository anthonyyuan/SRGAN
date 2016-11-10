require 'cutorch'

local M = { }

function M.parse(arg)
    local cmd = torch.CmdLine()

    cmd:text()
    cmd:text('Super resolution using perceptual loss, GAN, and residual network architecture')
    cmd:text('This is an implementation of the paper: Photo-Realistic Image Super-Resolution Using a Generative Adversarial Network (C. Ledig, 2016)')
    cmd:text()
    cmd:text('Options:')
    -- Global
    cmd:option('-manualSeed', 0,          'Manually set RNG seed')
    cmd:option('-nGPU',       1,          'Number of GPUs to use by default')
    cmd:option('-gpuid',      1,            'GPU id to use')
    cmd:option('-backend',    'cudnn',    'Options: cudnn | cunn')
    cmd:option('-cudnn',      'fastest',  'Options: fastest | default | deterministic')
    cmd:option('-gen',        'gen',      'Path to save generated files')
    cmd:option('-nThreads',    7,         'number of data loading threads')
	cmd:option('-save',       os.date("%Y-%m-%d_%H-%M-%S"),       'subdirectory to save/log experiments in')
    cmd:option('-defaultType', 'torch.FloatTensor', 'Default data type')
    -- Data
	cmd:option('-dataset', 'imagenet', 'dataset for training: imagenet | coco')
    cmd:option('-valset', 'Set5', 'validation set: val | Set5 | Set14 | B100 | Urban100')
    cmd:option('-sigma',    3,      'Sigma used for gaussian blur before shrinking an image.')
    cmd:option('-inter',    'bicubic', 'Interpolation method used for downsizing an image: bicubic | inter_area')
    -- Training
    cmd:option('-nEpochs',       0,       'Number of total epochs to run')
    cmd:option('-epochNumber',   1,       'Manual epoch number (useful on restarts)')
    cmd:option('-batchSize',     16,      'mini-batch size (1 = pure stochastic)')
    cmd:option('-patchSize',    96,         'Training patch size')
    cmd:option('-scale',        4,          'Super-resolution upscale factor')
    cmd:option('-testOnly',    'false', 'Run on validation set only')
    cmd:option('-printEvery',   1e2,       'Print log every # iterations')
    cmd:option('-testEvery',    1e3,       'Test every # iterations')
    cmd:option('-load',         '.',     'Load saved training model, history, etc.')
    -- Optimization
    cmd:option('-optimMethod',  'ADAM',  'Optimization method')
    cmd:option('-lr',         1e-4, 'initial learning rate')
	cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
	cmd:option('-momentum', 0.9, 'momentum (SGD only)')
	cmd:option('-beta1', 0.9, 'ADAM momentum')
	cmd:option('-type', 'cuda', 'type: float | cuda')
    -- Model
    cmd:option('-netType',      'ResNet', 'Generator network architecture. Options: ResNet | preResNet')
    cmd:option('-bottleneck',   'false',  'Use bottleneck architecture')
    cmd:option('-nResBlock',    15,     'Number of residual blocks in generator network')
    cmd:option('-nChannel',     3,      'Number of input image channels: 1 or 3')
    cmd:option('-nFeat',    64,     'Number of feature maps in residual blocks in generator network')
    cmd:option('-normalize',   'false',   'Normalize pixel values to be zero mean, unit std')
    cmd:option('-upsample',  'full',   'Upsampling method: full | bilinear')
    cmd:option('-filt_deconv',  3,      'filter size for deconvolution layer')
    cmd:option('-filt_recon',  17,      'filter size for reconstruction layer')
    -- Loss
    cmd:option('-abs',  0,  'L1 loss weight')
    cmd:option('-smoothL1', 0, 'Smooth L1 loss weight')
    cmd:option('-mse',   1,  'MSE loss weight')
    cmd:option('-perc',   0,  'VGG loss weight (perceptual loss)')
    cmd:option('-adv',   0,  'Adversarial loss weight')
    cmd:option('-tv',   0,  'Total variation regularization loss weight')
        -- VGG loss
    cmd:option('-vggDepth', '5-4', 'Depth of pre-trained VGG for use in perceptual loss')
        -- Adversarial loss
    cmd:option('-negval',       0.2,    'Negative value parameter for Leaky ReLU in discriminator network')
    cmd:option('-filtsizeD',    3,    'Filter size of stride 2 convolutions in discriminator network')

    ---------- Model options ----------------------------------
    cmd:option('-shareGradInput',  'false', 'Share gradInput tensors to reduce memory usage')
    cmd:option('-optnet',          'false', 'Use optnet to reduce memory usage')
    cmd:text()

    local opt = cmd:parse(arg or {})

    opt.sharedGradInput = opt.sharedGradInput=='true'
    opt.optnet = opt.optnet=='true'
    opt.bottleneck = opt.bottleneck=='true'
    opt.normalize = opt.normalize=='true'

    if opt.load ~= '.' then 
        opt.save = opt.load
        if not paths.dirp(paths.concat('../experiment',opt.save)) then
            print(opt.load .. ' does not exist. Train new model.')
            opt.load = false
        end
    else
        opt.load = false
    end

    opt.save = paths.concat('../experiment',opt.save)
    if not paths.dirp(opt.save) then
        paths.mkdir(opt.save)
        paths.mkdir(paths.concat(opt.save,'result'))
        paths.mkdir(paths.concat(opt.save,'model'))
    end

    torch.setdefaulttensortype('torch.FloatTensor')
    torch.setnumthreads(1)
    torch.manualSeed(opt.manualSeed)

    if opt.nGPU == 1 then
        cutorch.setDevice(opt.gpuid)
    end
    cutorch.manualSeedAll(opt.manualSeed)

    if opt.nEpochs < 1 then opt.nEpochs = math.huge end

    -- It is not recommended to change this option from default value ('val')
    if opt.valset ~= 'val' then
        opt.valset = 'benchmark/' .. opt.valset
    end

    torch.setdefaulttensortype(opt.defaultType)
    if opt.nGPU > 0 then
        opt.operateType = 'torch.CudaTensor'
    else
        opt.operateType = opt.defaultType
    end

	if opt.optimMethod == 'SGD' then
        opt.optimState = {
            method = optim.sgd,
			learningRate = opt.lr,
			weightDecay = opt.weightDecay,
			momentum = opt.momentum,
			dampening = 0,
			learningRateDecay = 1e-5,
			nesterov = true
		}
	elseif opt.optimMethod == 'ADADELTA' then
        opt.optimSate = {
            method = optim.adadelta,
			weightDecay = opt.weightDecay,
		}
	elseif opt.optimMethod == 'ADAM' then
		opt.optimState = {
            method = optim.adam,
			learningRate = opt.lr,
			beta1 = opt.beta1,
			weightDecay = opt.weightDecay
		}
	elseif opt.optimMethod == 'RMSPROP' then
		opt.optimState = {
            method = optim.rmsprop,
			learningRate = opt.lr,
		}
	else
		error('unknown optimization method')
	end  
    
    --print(opt)
    local opt_text = io.open(paths.concat(opt.save,'options.txt'),'a')
    opt_text:write(os.date("%Y-%m-%d_%H-%M-%S\n"))
    local function save_opt_text(key,value)
        if type(value) == 'table' then
            for k,v in pairs(value) do
                save_opt_text(k,v)
            end
        else
            if type(value) == 'function' then
                value = 'function'
            elseif type(value) == 'boolean' then
                value = value and 'true' or 'false'
            end
            opt_text:write(key .. ' : ' .. value .. '\n')
            return 
        end
    end
    save_opt_text(_,opt)
    opt_text:write('\n\n\n')
    opt_text:close()

    return opt
end

return M
