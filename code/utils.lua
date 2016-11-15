require 'nn'

local function deepCopy(tbl)
   -- creates a copy of a network with new modules and the same tensors
   local copy = {}
   for k, v in pairs(tbl) do
      if type(v) == 'table' then
         copy[k] = deepCopy(v)
      else
         copy[k] = v
      end
   end
   if torch.typename(tbl) then
      torch.setmetatable(copy, torch.typename(tbl))
   end
   return copy
end

local M = {}
local util = torch.class('util',M)

function util:__init(opt)
    self.opt = opt
    self.save = opt.save
end

function util:plot(tb,name)
    local fig = gnuplot.pdffigure(paths.concat(self.save,name .. '.pdf'))
	gnuplot.plot(name, torch.Tensor(tb), '-')
	gnuplot.grid(true)
	gnuplot.title(name)
	gnuplot.xlabel('iteration (*' .. self.opt.testEvery .. ')')
    if tb[1] < tb[#tb] then
        gnuplot.movelegend('right','bottom')
    else
        gnuplot.movelegend('right','top')
    end
	gnuplot.plotflush(fig)
	gnuplot.closeall()  
end

function util:store(model,loss,psnr)
    if torch.type(model) == 'nn.DataParallelTable' then
        model = model:get(1)
    end

    model = deepCopy(model):float():clearState()
    torch.save(paths.concat(self.save,'model','model_' .. #loss .. '.t7'),model)
    torch.save(paths.concat(self.save,'model','model_latest.t7'),model)
    torch.save(paths.concat(self.save,'loss.t7'),loss)
    torch.save(paths.concat(self.save,'psnr.t7'),psnr)
    torch.save(paths.concat(self.save,'opt.t7'),self.opt)
end

function util:load()
    local ok, loss, psnr
    if self.opt.load then
        ok,loss,psnr,opt = pcall( function()
                local loss = torch.load(paths.concat(self.save,'loss.t7'))
                local psnr = torch.load(paths.concat(self.save,'psnr.t7'))
                local opt = torch.load(paths.concat(self.save,'opt.t7'))
                return loss,psnr,opt
            end)
        if ok then
            print(('loaded history (%d epoch * %d iter/epoch)\n'):format(#loss,self.opt.testEvery))
        else
            print('history (loss, psnr, options) does not exist')
            loss, psnr = {},{}
        end
    else
        ok = false
        loss, psnr = {},{}
    end

    return ok, loss, psnr
end

return M.util