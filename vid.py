class PosVidConverter:
    def __init__(self):
        #deprecated
        # filePath = deviceId+".txt"
        # if(not exists(filePath)):
        #     print("Device does not exist")
        #     sys.exit()

        # contents = []
        # with open(filePath) as f:
        #      content = f.readlines()
        
        self.a=3510.605
        self.b=999.565
        self.c=-3.512
        self.pos1000Vid = 1.0
        self.pos1000 = 8.00 #change calibrated thru focus position 
        self.pos1400Vid = 1.4
        self.pos1400 = 10.00
        self.posMin = 0.5
        self.posMax = 12.482
        self.vidMin = 0.3
        self.vidInfPoint = 100
    
    def pos2vid(self, pos: float):
        if pos < self.posMin:
            pos = self.posMin
        if pos > self.posMax:
            pos = self.posMax

        vid = (self.a/(-0.001*(pos*1000 - self.pos1000) - self.c)-self.b)/1000 + self.pos1000Vid
        return vid

    def vid2pos(self, vid: float):
        if vid < self.vidMin:
            vid = self.vidMin
        if vid > self.vidInfPoint:
            vid = self.vidInfPoint

        x = (vid - self.pos1000Vid) * 1000
        y = -((self.a / (x + self.b)) + self.c)
        offValue = y / 0.001
        targetPos = self.pos1000 + offValue
        if (vid > self.vidInfPoint):
            targetPos = self.posMax
        return targetPos/1000


import numpy as np

converter = PosVidConverter()
print(converter.pos2vid(1.4))
diff0 = np.array([10, 60, 110, 70])
diff1 = np.array([converter.pos2vid(1.4+d0/1000)-converter.pos2vid(1.4) for d0 in diff0])
print(diff0, diff0.mean())
print(diff1, diff1.mean())

print()
print(-0.1, converter.pos2vid(1.3)-converter.pos2vid(1.4))
print(-0.2, converter.pos2vid(1.2)-converter.pos2vid(1.4))

