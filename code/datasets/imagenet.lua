local image = require 'image'
local paths = require 'paths'
local transform = require 'datasets/transforms'
local ffi = require 'ffi'
local cv = require 'cv'
require 'cv.imgcodecs'
require 'cv.imgproc'

local M = {}
local ImagenetDataset = torch.class('resnet.ImagenetDataset', M)

function ImagenetDataset:__init(opt, split)
    if split=='train' then
        local f
        if opt.trainset == 'train' then
            self.dir = '../dataset/ILSVRC2015/Data/CLS-LOC/train/'
            f = io.open('../dataset/ILSVRC2015/ImageSets/CLS-LOC/train_cls.txt')
        elseif opt.trainset == 'val' then
            self.dir = '../dataset/ILSVRC2015/Data/CLS-LOC/val/'
            f = io.open('../dataset/ILSVRC2015/ImageSets/CLS-LOC/val.txt')
        else
            error('wrong option: trainset')
        end
        print('\tloading train data...')
        self.imgPaths = {}
        while true do
            local line = f:read('*l')
            if not line then break end
            local imgPath = line:split(' ')[1] ..'.JPEG'
            self.imgPaths[#self.imgPaths+1] = imgPath
        end
        --[[
        --local cachePath = paths.concat('../dataset/cachePath_train.t7')
        --local cachePath = paths.concat('../dataset/cachePath_val.t7')
        if paths.filep(cachePath) then
            print('loading train data list from cache...')
            self.imgPaths = torch.load(cachePath)
        else
        --]]
            --torch.save(cachePath,self.imgPaths)
            --print('Saved trainset cache')
        --end
    elseif split=='val' then
        print('\tloading validation data...')
        if opt.valset == 'val' then
            local numVal = 10
            self.dir = '../dataset/ILSVRC2015/Data/CLS-LOC/val/'
            local rand = torch.randperm(50002)[{{1,numVal+2}}]
            self.imgPaths = {}
            local tmp = paths.dir(self.dir)
            for i = 1,rand:size(1) do
                if tmp[rand[i]] ~= '.' and tmp[rand[i]] ~= '..' then
                    self.imgPaths[i] = tmp[rand[i]]
                end
                if #self.imgPaths == numVal then break end
            end
        else
            self.dir = paths.concat('../dataset/',opt.valset)
            self.imgPaths = {}
            for f in paths.iterfiles(self.dir) do
                self.imgPaths[#self.imgPaths+1] = f
            end
        end
    end

    self.opt = opt
    self.split = split
end

function ImagenetDataset:get(i)
    local img = cv.imread{paths.concat(self.dir,self.imgPaths[i]), cv.IMREAD_COLOR}
    local h,w,_ = table.unpack(img:size():totable())
    local interpolation
    if self.opt.inter == 'bicubic' then
        interpolation = cv.bicubic
    elseif self.opt.inter == 'inter_area' then
        interpolation = cv.inter_area
    end

    --[[
    local input,target,target_blur
    if self.split == 'train' then
        local tps = self.opt.patchSize -- target patch size
        local ips = self.opt.patchSize / self.opt.scale -- input patch size
        if w-tps+1 < 1 or h-tps+1 < 1 then return end

        local x = torch.random(1, w-tps+1)
        local y = torch.random(1, h-tps+1)
        target = img[{{y,y+tps-1},{x,x+tps-1}}]
        target_blur = cv.GaussianBlur{src=target, ksize={3,3}, sigmaX = 0.95}
        input = cv.resize{target_blur, {ips,ips}, interpolation=cv.INTER_AREA}
    elseif self.split == 'val' then
        local hh,ww = self.opt.scale*math.floor(h/self.opt.scale), self.opt.scale*math.floor(w/self.opt.scale)
        target = img[{{1,hh},{1,ww}}]
        local hhi,wwi = hh/self.opt.scale, ww/self.opt.scale
        target_blur = cv.gaussianblur{src=target, ksize={3,3}, sigmax = 0.95}
        input = cv.resize{target_blur, {wwi,hhi}, interpolation=cv.inter_area}
    end
    --]]
    local hh,ww = self.opt.scale*math.floor(h/self.opt.scale), self.opt.scale*math.floor(w/self.opt.scale)
    local target = img[{{1,hh},{1,ww}}]
    local hhi,wwi = hh/self.opt.scale, ww/self.opt.scale
    -- Resizing an entire image is inefficient, but it gives better quality than patch-wise computation.
    local target_blur = cv.GaussianBlur{src=target, ksize={3,3}, sigmaX = self.opt.sigma}
    local input = cv.resize{target_blur, {wwi,hhi}, interpolation=interpolation}

    if self.split == 'train' then 
        local tps = self.opt.patchSize -- target patch size
        local ips = self.opt.patchSize / self.opt.scale -- input patch size
        if ww-tps+1 < 1 or hh-tps+1 < 1 then return end

        local ix = torch.random(1, wwi-ips+1)
        local iy = torch.random(1, hhi-ips+1)
        local tx = self.opt.scale*(ix-1)+1
        local ty = self.opt.scale*(iy-1)+1

        input = input[{{iy,iy+ips-1},{ix,ix+ips-1}}]
        target = target[{{ty,ty+tps-1},{tx,tx+tps-1}}]
    end

    input = input:permute(3,1,2):index(1,torch.LongTensor{3,2,1}):float()/255
    target = target:permute(3,1,2):index(1,torch.LongTensor{3,2,1}):float()/255

    return {
        input = input,
        target = target
    }
end

function ImagenetDataset:size()
    --print(#self.imgPaths)
    return #self.imgPaths
end

-- Computed from random subset of ImageNet training images
local meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}
local pca = {
   eigval = torch.Tensor{ 0.2175, 0.0188, 0.0045 },
   eigvec = torch.Tensor{
      { -0.5675,  0.7192,  0.4009 },
      { -0.5808, -0.0045, -0.8140 },
      { -0.5836, -0.6948,  0.4203 },
   },
}

function ImagenetDataset:augment()
    if self.split == 'train' then
        return transform.Compose{
            transform.ColorJitter({
                brightness = 0.1,
                contrast = 0.1,
                saturation = 0.1
            }),
            --transform.Lighting(0.1, pca.eigval, pca.eigvec),
            transform.HorizontalFlip(0.5),
            transform.Rotation(1)
        }
    elseif self.split == 'val' then
        return function(sample) return sample end
    end
end

return M.ImagenetDataset
