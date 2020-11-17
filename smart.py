# USAGE
# python3 smart.py

# Importing Libraries
import RPi.GPIO as GPIO
import time
from tkinter import *
import tkinter.font
from PIL import Image, ImageTk
from tkinter import messagebox, Label, Button, FALSE, Tk, Entry

username = ("Mira")
password = ("1234")

# Raspberry Pi 3 Pin Settings
# tkinter GUI basic settings
Gui = Tk()
Gui.title("GUI in Raspberry Pi 3")
Gui.config(background= "#0080FF")
Gui.minsize(700,300)
Font1 = tkinter.font.Font(family = 'Helvetica', size = 24, weight = 'bold')

def front():
        
    Text1 = Label(Gui,text=' front ', font = Font1, bg = '#0080FF', fg='green', padx = 0)
    Text1.grid(row=0,column=2)
    # setup the GPIO pin for the servo
    servo_pin = 13
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(servo_pin,GPIO.OUT)
    # setup PWM process
    pwm = GPIO.PWM(servo_pin,50) # 50 Hz (20 ms PWM period)
    pwm.start(0) # start PWM by rotating to 90 degrees
    for ii in range(1):
        pwm.ChangeDutyCycle(12.0) # rotate to 180 degrees
        time.sleep(0.5)

    pwm.ChangeDutyCycle(0) # this prevents jitter
    pwm.stop() # stops the pwm on 13
    GPIO.cleanup() # good practice when finished using a pin
        
def back():
        
    Text1 = Label(Gui,text=' back ', font = Font1, bg = '#0080FF', fg='green', padx = 0)
    Text1.grid(row=0,column=2)
    # setup the GPIO pin for the servo
    servo_pin = 13
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(servo_pin,GPIO.OUT)

    # setup PWM process
    pwm = GPIO.PWM(servo_pin,50) # 50 Hz (20 ms PWM period)

    pwm.start(0) # start PWM by rotating to 90 degrees

    for ii in range(1):
        pwm.ChangeDutyCycle(2.0) # rotate to 0 degrees
        time.sleep(0.5)

    pwm.ChangeDutyCycle(0) # this prevents jitter
    pwm.stop() # stops the pwm on 13
    GPIO.cleanup() # good practice when finished using a pin

def pin():
    
    # Funtion for Buttons ended here
    Text1 = Label(Gui,text='Status:', font = Font1, fg='#FFFFFF', bg = '#0080FF', padx = 50, pady = 50)
    Text1.grid(row=0,column=0)

    Button1 = Button(Gui, text='FRONT', font = Font1, command = front, bg='bisque2', height = 1, width = 10)
    Button1.grid(row=1,column=0)

    Button4 = Button(Gui, text='BACK', font = Font1, command = back, bg='bisque2', height = 1, width = 10)
    Button4.grid(row=1,column=2)

    Text3 = Label(Gui,text='Smart Surveillance monitoring system', font = Font1, bg = '#0080FF', fg='#FFFFFF', padx = 50, pady = 50)
    Text3.grid(row=2,columnspan=5)
 
# auth simple GUI created
def try_login():
    print("Trying to login...")
    if username_guess.get() == username and password_guess.get() == password:
        messagebox.showinfo("-- COMPLETE --", "You Have Now Logged In.", icon="info")
        pin()
    
    else:
        messagebox.showinfo("-- ERROR --", "Please enter valid infomation!", icon="warning")


GPIO.cleanup() # good practice when finished using a pin
#Gui Things
window = Tk()
window.resizable(width=FALSE, height=FALSE)
window.title("Log-In")
window.geometry("300x200")

#Creating the username & password entry boxes
username_text = Label(window, text="Username:")
username_guess = Entry(window)
password_text = Label(window, text="Password:")
password_guess = Entry(window, show="*")

#attempt to login button
attempt_login = Button(window,text="Login", command=try_login)

username_text.pack()
username_guess.pack()
password_text.pack()
password_guess.pack()
attempt_login.pack()

#Main Starter
window.mainloop()
GPIO.cleanup() 
