import wx
from wx.lib.wxcairo import BitmapFromImageSurface
import cairo

class Frame(wx.Frame):
    '''straight inherited from wx.Frame, but created as a cutout in case
    I need to customize it later.'''

    pass

class MainFrame(Frame):
    '''The main viewer Frame.
    '''

    def __init__(self, *args, **kwargs):

        Frame.__init__(self, *args, **kwargs)

        self.createPanel()
        self.createMap()

        self.panel.Layout()

    def createPanel(self, *args, **kwargs):

        self.panel = wx.Panel(self)
        self.sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.panel.SetSizer(self.sizer)
        #self.panel.SetBackgroundColour('BLUE')

    def createMap(self, filename='output/mich2b.png', *args, **kwargs):

        self.mapper = Mapper()

        self.disp = wx.StaticBitmap(parent=self.panel)
        self.disp.Create(parent=self.panel, bitmap=self.mapper.bitmap)
        self.sizer.Add(self.disp)

class Mapper(object):
    """
    Draws isocontours on a map
    """

    def __init__(self, filename='output/mich2b.png', *args, **kwargs):

        with open(filename) as inf:
            self.img = cairo.ImageSurface.create_from_png(inf)

    @property
    def bitmap(self):
        return BitmapFromImageSurface(self.img)
