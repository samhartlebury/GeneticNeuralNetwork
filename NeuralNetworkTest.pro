QT -= gui

CONFIG += c++11 console
CONFIG -= app_bundle

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

INCLUDEPATH += \
            $$(NXLIBS)\Support\OpenCV4_Qt\include

DEPENDPATH += \
            $$(NXLIBS)\Support\OpenCV4_Qt\lib

SOURCES += \
        main.cpp \
        neuralnetwork.cpp

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

HEADERS += \
    neuralnetwork.h


win32:CONFIG(release, debug|release): LIBS += -L$$(NXLIBS)/Support/OpenCV4_Qt/lib/    -lopencv_core430
else:win32:CONFIG(debug, debug|release): LIBS += -L$$(NXLIBS)/Support/OpenCV4_Qt/lib/ -lopencv_core430d
win32:CONFIG(release, debug|release): LIBS += -L$$(NXLIBS)/Support/OpenCV4_Qt/lib/    -lopencv_highgui430
else:win32:CONFIG(debug, debug|release): LIBS += -L$$(NXLIBS)/Support/OpenCV4_Qt/lib/ -lopencv_highgui430d
win32:CONFIG(release, debug|release): LIBS += -L$$(NXLIBS)/Support/OpenCV4_Qt/lib/    -lopencv_imgproc430
else:win32:CONFIG(debug, debug|release): LIBS += -L$$(NXLIBS)/Support/OpenCV4_Qt/lib/ -lopencv_imgproc430d
win32:CONFIG(release, debug|release): LIBS += -L$$(NXLIBS)/Support/OpenCV4_Qt/lib/    -lopencv_imgcodecs430
else:win32:CONFIG(debug, debug|release): LIBS += -L$$(NXLIBS)/Support/OpenCV4_Qt/lib/ -lopencv_imgcodecs430d
win32:CONFIG(release, debug|release): LIBS += -L$$(NXLIBS)/Support/OpenCV4_Qt/lib/    -lopencv_features2d430
else:win32:CONFIG(debug, debug|release): LIBS += -L$$(NXLIBS)/Support/OpenCV4_Qt/lib/ -lopencv_features2d430d
