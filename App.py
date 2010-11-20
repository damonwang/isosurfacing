#!/usr/bin/env python

import wx
from MainFrame import MainFrame

class App(wx.App):
    def OnInit(self):
        self.mainframe = MainFrame(parent=None, id=-1, title='Isocontours')
        self.SetTopWindow(self.mainframe)
        self.mainframe.Show()
        return True

