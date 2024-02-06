import math
import numpy as np
from pynput.keyboard import Controller,Key
import cv2
import cvzone
import mediapipe as mp
import HandTrackingModule as htm

cap=cv2.VideoCapture(0)
keyboard=Controller()
# cap.set(3,1280)
# cap.set(4,720)
new_food="strawberry (1).png"
detector=htm.HandDetector(detectionCon=0.8)
class SnakeGameClass:
    def __init__(self):
        self.food="orange"
        self.points=[]# all points of the snake
        self.lengths=[]# distance between each point
        self.currentLength=0# total length of the snake
        self.allowedLength=250#Total allowed length
        self.previousHead=0,0#previous head point
        self.foodPoint=0,0
        self.score=0
        self.GameOver=False
        self.randomFoodLocation()



    def randomFoodLocation(self):
        self.foodPoint=np.random.randint(100, 900),np.random.randint(150, 600)

    def update(self,imgMain,CurrentHead,pathFood,food):
        self.imgFood = cv2.imread(pathFood, cv2.IMREAD_UNCHANGED)
        self.hFood, self.wFood, _ = self.imgFood.shape
        if self.GameOver:
            cvzone.putTextRect(imgMain,"Game Over",[200,400],scale=5,
                               thickness=4,offset=10)
            cvzone.putTextRect(imgMain, f'Your Score: {self.score}', [200, 500], scale=5,
                               thickness=4, offset=10)
            cvzone.putTextRect(imgMain, "fold your fingers to restart", [200, 580], scale=3, thickness=3,
                                offset=3)
            # self.score=0
        else:
            px,py=self.previousHead
            cx,cy=CurrentHead

            self.points.append([cx,cy])
            distance=math.hypot(cx-px,cy-py)
            self.lengths.append(distance)
            self.currentLength+=distance
            self.previousHead=cx,cy
            # Length Reduction
            if self.currentLength>self.allowedLength:
                for i,length in enumerate(self.lengths):
                    self.currentLength-=length
                    self.lengths.pop(i)
                    self.points.pop(i)
                    if self.currentLength<self.allowedLength:
                        break
            #check if snake eat food
            rx,ry=self.foodPoint
            if rx-self.wFood//2<cx<rx+self.wFood//2 and ry-self.hFood//2<cy<ry+self.hFood//2:
                if self.score%2==0:
                    game.pathFood=new_food
                self.randomFoodLocation()
                if food=="strawberry":
                    game.food="strawberry"
                    self.allowedLength += 100
                    self.score += 4
                else:
                    game.food="orange"
                    self.allowedLength+=50
                    self.score+=1



            #Draw Snake
            if self.points:
                for i,point in enumerate(self.points):
                    if i!=0:
                        cv2.line(imgMain,self.points[i-1],self.points[i],(0,0,255),20)
                cv2.circle(imgMain,self.points[-1],10,(0,126,255),cv2.FILLED)

            #Draw Food
            rx,ry=self.foodPoint
            imgMain=overlay(imgMain,self.imgFood,rx,ry)
            # check if snake eat food
            rx, ry = self.foodPoint
            if rx - self.wFood // 2 < cx < rx + self.wFood // 2 and ry - self.hFood // 2 < cy < ry + self.hFood // 2:
                self.randomFoodLocation()
                self.allowedLength += 50
                self.score += 1
                print(self.score)
            cvzone.putTextRect(imgMain,f'Score: {self.score}',[50,80],scale=5,thickness=3,colorR=(0,255,255),colorT=(0,0,0),offset=5)
            # Check for collision
            pts = np.array(self.points[:-2], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(imgMain, [pts], False, (255, 255, 255), 3)
            minDist = cv2.pointPolygonTest(pts, (cx, cy), True)
            # print(minDist)
            if -1 <= minDist <= 1:
                # print("Hit")
                self.GameOver = True
                self.points = []  # all points of the snake
                self.lengths = []  # distance between each point
                self.currentLength = 0  # total length of the snake
                self.allowedLength = 150  # Total allowed length
                self.previousHead = 0, 0
                self.randomFoodLocation()
        return imgMain
# orange=cv2.imread("close-up-ripe-strawberry-isolated_299651-121.jpg")
# orange=cv2.resize(orange,(100,100))
# cv2.imwrite("strawberry.png",orange)

game=SnakeGameClass()

def overlay(img,apple,rx,ry):
    # Get the region of interest for placing the balloon

    # Ensure fg_img has an alpha channel
    apple = cv2.cvtColor(apple, cv2.COLOR_BGR2BGRA)
    cx,cy=(rx-game.wFood//2,ry-game.hFood//2)
    roi = img[cy:cy + apple.shape[0], cx:cx + apple.shape[1]]

    # Resize the balloon to match the roi size (explicit rounding to integers)
    apple_resized = cv2.resize(apple, (int(roi.shape[1]), int(roi.shape[0])))

    # Create a mask from the alpha channel of the resized balloon
    mask = apple_resized[:, :, 3]

    # Invert the mask to use it as an alpha channel
    mask = ~mask

    # Resize the inverted mask to match the balloon size
    mask = cv2.resize(mask.astype(np.uint8), (apple.shape[1], apple.shape[0]))

    alpha = 1.0 - mask.astype(float) / 255.0

    # Split the balloon and alpha channel
    apple_rgb = apple_resized[:, :, :3]

    # Merge the RGB channels with the inverted alpha channel
    result = cv2.merge([roi[:, :, 0] * (1.0 - alpha) + apple_rgb[:, :, 0] * alpha,
                        roi[:, :, 1] * (1.0 - alpha) + apple_rgb[:, :, 1] * alpha,
                        roi[:, :, 2] * (1.0 - alpha) + apple_rgb[:, :, 2] * alpha])

    # Place the result back into the original image
    img[cy:cy + apple_resized.shape[0], cx:cx + apple_resized.shape[1]] = result
    return img
counter=0
while True:
    success,img=cap.read()
    img=cv2.resize(img,(1100,720))
    img=cv2.flip(img,1)
    hands,_=detector.findHands(img,draw=False)
    if hands:
        lmList=hands[0]['lmList']
        pointIndex=lmList[8][:2]
        # print(game.score)
        if game.score%4==0 and game.score!=0 and game.food!='strawberry':
            food = "strawberry"
            img=game.update(img,pointIndex,new_food,food)
        else:
            food="orange"
            img=game.update(img,pointIndex,"small_orange.png",food)
        # cv2.circle(img,pointIndex,10,(255,0,255),cv2.FILLED)
        sm = sum(detector.fingersUp(hands[0]['lmList']))
        print(sm)
        if sm == 1:
            keyboard.press('r')
    counter+=1
    cv2.imshow("Snake Game",img)
    key=cv2.waitKey(1)
    if key==ord('r'):
        game.score=0
        game.GameOver=False