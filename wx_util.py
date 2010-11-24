import wx

def createMenuBar(menubar, setInto=None):
    '''creates a menubar and optionally attaches it to something

    a menubar is a list of menus.

    Args:
        setInto: something with a SetMenuBar method
        menubar: see above

    Returns: a wx.Menubar'''

    rv = wx.MenuBar()
    for menu in menubar:
        m = createMenu(menu=menu, container=setInto)
        rv.Append(menu=m, title=menu[0])

    if setInto is not None:
        setInto.SetMenuBar(rv)

    return rv

def createMenu(menu, container, setInto=None):
    '''creates a menu and optionally attaches it to a wx.MenuBar

    a menu is described by a 2-tuple whose
        first element is a title suitable for wx.Menu()
        second element is a list of tuples and dicts:
            lists represent submenus
        dicts represent menu items

    a menu item is represented by a dict suitable for **kwargs expansion into
    wx.Menu.Append().

    Args:
        menu: see above
        setInto: a wx.MenuBar to attach the result to

    Returns: a wx.Menu'''

    rv = wx.Menu()
    for item in menu[1]:
        if isinstance(item, tuple):
            rv.AppendMenu(text=item[0], submenu=createMenu(item))
        elif isinstance(item, dict):
            handler = item["handler"]
            del item["handler"]
            m = rv.Append(**item)
            container.Bind(event=wx.EVT_MENU, handler=handler, source=m)
        else:
            raise TypeError(item)

    if setInto is not None:
        setInto.Append(menu=rv, title=menu[0])

    return rv
