from __future__ import with_statement

import wx
from wx.lib.wxcairo import BitmapFromImageSurface
import cairo
from cairo import ImageSurface, Context, SurfacePattern, SolidPattern
import wx_util

if __debug__:
    import cnv
else: import cy_cnv as cnv

class Frame(wx.Frame):
    '''straight inherited from wx.Frame, but created as a cutout in case
    I need to customize it later.'''

    pass



class MainFrame(Frame):
    '''The main viewer Frame.
    '''

    def __init__(self, *args, **kwargs):

        Frame.__init__(self, *args, **kwargs)

        self.mapper = None

        self.createPanel()
        self.createMenuBar()

        self.panel.Layout()

    def createPanel(self, *args, **kwargs):

        self.panel = wx.Panel(self)
        self.sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.panel.SetSizer(self.sizer)
        #self.panel.SetBackgroundColour('BLUE')

    def createMap(self, filename='output/mich2b.png', *args, **kwargs):

        try:
            self.mapper = Mapper(filename=filename, *args, **kwargs)
        except BaseException:
            self.mapper = None
            raise

        self.disp = wx.StaticBitmap(parent=self.panel)
        self.disp.Create(parent=self.panel, bitmap=self.mapper.bitmap)
        self.sizer.Add(self.disp)

    def createMenuBar(self, *args, **kwargs):

        filemenu = [ dict(id=-1, text="&Open", help="Open a data file", handler=self.OnClickFileOpen),
                dict(id=-1, text="&Quit", help="Quit the application", handler=self.OnClickFileQuit) ]

        self.menubar = wx_util.createMenuBar(setInto=self, menubar=[("&File", filemenu)])

    def refreshBitmap(self):

        self.disp.SetBitmap(self.mapper.bitmap)
        self.Refresh()

    def OnClickFileOpen(self, event):

        dlg = wx.FileDialog(parent=self, message="Choose a map",
                defaultDir=os.getcwd(), style=wx.OPEN | wx.CHANGE_DIR)
        if dlg.ShowModal() == wx.ID_OK:
            try:
                self.createMap(filename=dlg.GetPath())
            except IOError, e:
                wx.MessageBox('Could not open file "%s":\n\t[Errno %d] %s' % (e.filename, e.errno, e.strerror))
        dlg.Destroy()

        self.sizer.SetSizeHints(self)
        self.sizer.Layout()

    def OnClickFileQuit(self, event):

        self.Close()

class Mapper(object):
    """
    Draws isocontours on a map

    """

    def __init__(self, filename='output/mich2b.png', *args, **kwargs):

        with open(filename) as inf:
            self.img = ImageSurface.create_from_png(inf)
        # TODO find a better way to copy this
        with open(filename) as inf:
            self.bg_pat = SurfacePattern(ImageSurface.create_from_png(inf))
        self.ctx = Context(self.img)
        self.cache = {}
        self.data = cnv.png_to_ndarray(filename)
        self.aspect = (1., 1.)
        self.minmax = cnv.make_minmax(self.data)
        self.visible = set()

        self.pencolor = kwargs.get('pencolor', (0., 0., 0.))

    def get_isocontour(self, isovalue):

        try: return self.cache[isovalue]
        except KeyError:
            ctx = Context(ImageSurface(cairo.FORMAT_A1, self.img.get_width(), self.img.get_height()))
            for (x1, y1), (x2, y2), val in cnv.param_isocontours(self.data, self.aspect, self.minmax, 1, isovalue):
                ctx.move_to(x1, y1)
                ctx.line_to(x2, y2)
                ctx.stroke()
            mask = ctx.get_target()
            self.cache[isovalue] = mask
            return mask

    def draw(self, isovalue):

        if isovalue not in self.visible:
            self.set_source_rgba(*self.pencolor)
            self.ctx.mask_surface(self.get_isocontour(isovalue))
            self.visible.add(isovalue)

    def erase(self, isovalue):

        if isovalue in self.visible:
            self.set_source(self.bg_pat)
            self.ctx.mask_surface(self.get_isocontour(isovalue))
            self.visible.remove(isovalue)

    @property
    def bitmap(self):
        return BitmapFromImageSurface(self.img)
