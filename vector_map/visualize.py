import threading

from sympy import Segment
from tkinter import Tk, Canvas, BOTH, Label, FIRST, LAST

from .geometric_map import Region

root = None
def main():
    global root
    root = Tk()
    root.title("Simulation Space")
    root.geometry("1000x1300")
    root.mainloop()

def init_visualize():
    threading.Thread(target=main).start()

WINDOW_X = 1000
WINDOW_Y = 1300
class SimulationSpace:

    def __init__(self, region:Region, window_x=WINDOW_X, window_y=WINDOW_Y):
        global root
        for widgets in root.winfo_children():
            widgets.destroy()
        self.window_x = window_x
        self.window_y = window_y
        self.root = root
        canvas = Canvas(root, width=1000, height=1250)
        self.canvas = canvas
        self.world = region.world
        self.origin = region.world.map.raster.origin
        first = True
        for b in region.outer_boundary:
            l = b.segment
            x = l.p1.x
            y = l.p1.y
            if first:
                first = False
                max_x = x
                min_x = x
                max_y = y
                min_y = y
            else:
                if x > max_x: max_x = x
                elif x < min_x: min_x = x
                if y > max_y: max_y = y
                elif y < min_y: min_y = y
            self.offset_x = min_x + 1.5 
            self.offset_y = min_y + 1.9 
            range_x = max_x - min_x
            range_y = max_y - min_y
            scale_x = WINDOW_X / (range_x + 1)
            scale_y = WINDOW_Y / (range_y + 1)
            if scale_x > scale_y: self.scale = scale_y 
            else: self.scale = scale_x
            self.total_y = range_y * self.scale
        for b in region.outer_boundary:
            self.draw_line(b.segment)
        self.canvas.pack()
        self.callback = {}
        self.mouse = MouseDriver(self)
    
    def coord_to_pix(self, *loc):
        if len(loc) == 1:
            val = loc[0]
            if isinstance(val, tuple):
                x = val[0]
                y = val[1]
            else:
                x = val.x
                y = val.y
        else:
            x = loc[0]
            y = loc[1]

        pix_x = int((x - self.offset_x) * self.scale) 
        pix_y = int((-y + self.offset_y) * self.scale + self.total_y)
        pix_x -= self.origin[0]
        pix_y += self.origin[1]
        return pix_x, pix_y

    def pix_to_coord(self, *pix):
        if len(pix) == 1:
            val = pix[0]
            if isinstance(val, tuple):
                x = float(val[0])
                y = float(val[1])
            else:
                x = float(val.x)
                y = float(val.y)
        else:
            x = float(pix[0])
            y = float(pix[1])
            x -= self.origin[0] 
            y -= self.origin[1]
        loc_x = x / self.scale + self.offset_x #+ self.origin[0]
        loc_y = (self.total_y - y) / self.scale + self.offset_y #+ self.origin[1]
        return loc_x, loc_y
    
    def draw_line(self, line:Segment, arrow=None):
        x1, y1 = self.coord_to_pix(line.p1)
        x2, y2 = self.coord_to_pix(line.p2)
        if arrow:
            if arrow == "first": arrow = FIRST
            elif arrow == "last": arrow = LAST
            else: arrow = None
        self.canvas.create_line(x1, y1, x2, y2, arrow=arrow)

    def start_mouse(self):
        self.mouse = MouseDriver(self)
    
    def set_object(self, loc):
        callback = self.callback.get('set_object')
        if callback:
            callback(*(self.pix_to_coord(loc)))

    def set_callback(self, type, callback):
        self.callback[type] = callback
    
    def create_circle(self, x, y, d, color="black"):
        p1 = self.coord_to_pix((x-d), (y-d))
        p2 = self.coord_to_pix((x+2*d, (y+2*d)))
        self.canvas.create_oval(*p1, *p2, outline=color)
    
    def create_mark(self, x, y, mark='x', color='black'):
        p = self.coord_to_pix(x, y)
        self.canvas.create_text(*p, text=mark, fill=color)

    def loop(self):
        self.root.mainloop()

class MouseDriver:
    def __init__(self, master) -> None:
        self.master = master
        label = Label(
            master.root,
            text="mouse position",
            width=30,
            height=5,
        )
        self.label = label
        master.canvas.bind("<Motion>", self.on_mouse_move)
        master.canvas.bind("<Button>", self.on_mouse_click)
        label.pack()
        self.mode = 'set'
    
    def set_mode(self, mode):
        self.mode = mode
    
    def on_mouse_move(self, event):
        p = self.master.pix_to_coord(event)
        x = p[0]
        y = p[1]
        self.label["text"] = f'mouse x:{x:.2f}m, y:{y:.2f}m'
    
    def on_mouse_click(self, event):
        if self.mode == 'set':
            if event.num != 1: return
            self.master.canvas.create_text(event.x, event.y, text='x')
            self.master.set_object(event)
