require 'torch'
require 'paths'
require 'optim'
require 'nn'
require 'cutorch'
require 'gnuplot'

local opts = require 'opts'
local opt = opts.parse(arg)

local util = require 'utils'(opt)
local load, loss, psnr = util:load()

local DataLoader = require 'dataloader'
local Trainer = require 'train'

print('loading model and criterion...')
local model = require 'model/init'(opt)
local criterion = require 'loss/init'(opt)

print('Creating data loader...')
local trainLoader, valLoader = DataLoader.create(opt)
local trainer = Trainer(model, criterion, opt, optimState)

print('Train start')
local startEpoch = load and #loss+1 or opt.epochNumber
for epoch = startEpoch, opt.nEpochs do
    local lossG,lossD = trainer:train(epoch, trainLoader)
    local psnr_ = trainer:test(epoch, valLoader)

    loss[#loss+1] = lossG
    psnr[#psnr+1] = psnr_

    util:plot(loss,'loss')
    util:plot(psnr,'PSNR')

    util:store(model,loss,psnr)
end