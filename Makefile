
CC = g++
TARGET = demo
SOURCE = main.cpp
CFLAGS = -Wall

OPENCV_INCLUDE_PATH = /usr/local/Cellar/opencv/2.4.13/include/
OPENCV_LINK_PATH = /usr/local/lib/
STK_INCLUDE_PATH = /usr/local/include/stk/
STK_LINK_PATH = /usr/local/lib/

INCLUDES = -I$(OPENCV_INCLUDE_PATH) -I$(STK_INCLUDE_PATH)
LFLAGS = -L$(OPENCV_LINK_PATH) -L$(STK_LINK_PATH)
LIBS = -lopencv_highgui -lopencv_core -lopencv_video -lopencv_videostab -lopencv_imgproc -lstk -lpthread
MACFLAG = -D__MACOSX_CORE__
FRAMEWORKS = -framework CoreAudio -framework CoreMIDI -framework CoreFoundation

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CC) $(CFLAGS) $(INCLUDES) $(MACFLAG) -o $(TARGET) $(SOURCE) $(LFLAGS) $(LIBS) $(FRAMEWORKS)

clean:
	$(RM) *.o *~ $(TARGET)
