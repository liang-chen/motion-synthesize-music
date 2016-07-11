
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/video.hpp"
#include "opencv2/highgui.hpp"

#include "RtAudio.h"

#include <iostream>
#include <math.h>

using namespace std;

#define FRAME_WIDTH 640
#define FRAME_HEIGHT 480

const double PI     = 3.14159265359;
const double TWO_PI = 2 * PI;
double frequency_global = 440.0;
double phase_global = 0;
double mag_global = 0;

static int sin( void *outputBuffer, void *, unsigned int nBufferFrames,
         double, RtAudioStreamStatus, void * )
{
  // Cast the buffer to the correct data type.
  double *my_data = (double *) outputBuffer;
  const int sampleRate = 44100;
  const double samplePeriod = 1.0 / sampleRate;

  // We know we only have 1 sample per frame here.
  for ( int i=0; i<nBufferFrames; i++ ) {
    const double phaseIncrement = TWO_PI * frequency_global * samplePeriod;
    my_data[i] = mag_global * sin( phase_global );
    phase_global += phaseIncrement;
    if ( phase_global > TWO_PI ) phase_global -= TWO_PI;
  }

  return 0;
}

static void help()
{
    cout <<
            "\nThis program demonstrates dense optical flow algorithm by Gunnar Farneback\n"
            "Mainly the function: calcOpticalFlowFarneback()\n"
            "Call:\n"
            "./fback\n"
            "This reads from video camera 0\n" << endl;
}

static void drawOptFlowMap(const cv::Mat& flow, cv::Mat& cflowmap, int step,
                    double, const cv::Scalar& color)
{
    for(int y = 0; y < cflowmap.rows; y += step)
        for(int x = 0; x < cflowmap.cols; x += step)
        {
            const cv::Point2f& fxy = flow.at<cv::Point2f>(y, x);
            cv::line(cflowmap, cv::Point(x,y), cv::Point(cvRound(x+fxy.x), cvRound(y+fxy.y)),
                 color);
            cv::circle(cflowmap, cv::Point(x,y), 2, color, -1);
        }
}

typedef struct{
      cv::Point left;
      cv::Point rite;
      float log_prob;
} TRACK_STATE;

static void InitTrackState(TRACK_STATE& state, int rows, int cols){
      state.log_prob = 0;
      state.left.x = 1;
      state.rite.x = cols - 1;
      state.left.y = rows/2;
      state.rite.y = rows/2;
}

static void TrackHands(const cv::Mat& flow, TRACK_STATE& obj_state){
      float thresh = 10;
      cv::Point left = obj_state.left;
      cv::Point rite = obj_state.rite;
      vector<cv::Point2f> salient_points;
      vector<int> labels;
      
      for(int y = 0; y < flow.rows; y++){
            for(int x = 0; x < flow.cols; x++){
                  if(cv::norm(flow.at<cv::Point2f>(y,x)) > thresh){
                        salient_points.push_back(cv::Point2f(x,y));
                        labels.push_back(norm(cv::Point(x,y) - left) > norm(cv::Point(x,y) - rite)?1:0);
                  }
            }
      }
      
      if(salient_points.size() < 2){
            obj_state.log_prob = 0;
            return;
      }
      
      cv::Mat mPoints(salient_points.size() ,1,CV_32FC2, &salient_points.front());
      cv::Mat mLabels(labels.size() ,1,CV_16U, &labels.front());
      cv::Mat centers;
      
      cv::kmeans(mPoints, 2, mLabels,
            cv::TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 3, 0.001),
               3, cv::KMEANS_PP_CENTERS, centers);
      
      obj_state.left = centers.at<cv::Point2f>(0);
      obj_state.rite = centers.at<cv::Point2f>(1);
      
      //cout<< obj_state.left.x << obj_state.left.y<<endl;
      
      float log_prob = 0;
      
      for(int i = 0; i < mPoints.rows; i++){
            const cv::Point& loc = mPoints.at<cv::Point2f>(i);
            const cv::Point2f& fxy = flow.at<cv::Point2f>(loc.y, loc.x);
            log_prob += (cv::norm(fxy) - 1.5*thresh)/10000;
      }
      //cout <<"here "<<obj_state.log_prob<< " "<< (0.8*expf(obj_state.log_prob) + 0.2) << " "<< (0.2*(1-expf(obj_state.log_prob)) + 0.8)<< endl;
      log_prob += logf( (0.8*expf(obj_state.log_prob) + 0.2) / (0.2*expf(obj_state.log_prob) + 0.8) );
      //cout<<log_prob<<endl;
      obj_state.log_prob = log_prob;
}

int main(int argc, char** argv)
{
    //cv
    cv::CommandLineParser parser(argc, argv, "{help h||}");
    cv::VideoCapture cap(0);
    help();
    if( !cap.isOpened() )
        return -1;

    cv::Mat flow, cflow, frame, gray, prevgray, uflow;
    cap.set(CV_CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);
    cv::namedWindow("flow", 1);
    
    //audio
    unsigned int nBufferFrames = 256;  // 256 sample frames
    unsigned int sampleRate = 44100;
    unsigned int nChannels = 1;
    RtAudio dac;

    // Open the default realtime output device.
    RtAudio::StreamParameters parameters;
    parameters.deviceId = dac.getDefaultOutputDevice();
    parameters.nChannels = nChannels;
    try {
      dac.openStream( &parameters, NULL, RTAUDIO_FLOAT64, sampleRate, &nBufferFrames, &sin );
    }
    catch ( RtAudioError &error ) {
      error.printMessage();
      exit( EXIT_FAILURE );
    }

    try {
      dac.startStream();
    }
    catch ( RtAudioError &error ) {
      error.printMessage();
      exit( EXIT_FAILURE );
    }

      
    TRACK_STATE obj_state;
    InitTrackState(obj_state, FRAME_HEIGHT, FRAME_WIDTH);
    for(;;)
    {
        cap >> frame;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        if( !prevgray.empty() )
        {
            cv::calcOpticalFlowFarneback(prevgray, gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
            cv::cvtColor(prevgray, cflow, cv::COLOR_GRAY2BGR);
            TrackHands(flow, obj_state);
            if(obj_state.log_prob > 0){
                  mag_global = cv::norm(obj_state.left.x - obj_state.rite.x)/50;
                  frequency_global = 440*(0.5 + cv::norm(obj_state.left.y - obj_state.rite.y)/80);
            }
            else{
                  //InitTrackState(obj_state, FRAME_HEIGHT, FRAME_WIDTH);
            }
            cv::circle(cflow, obj_state.left, 20, cv::Scalar(0, 255, 0), -1);
            cv::circle(cflow, obj_state.rite, 20, cv::Scalar(0, 255, 0), -1);
            imshow("flow", cflow);
        }
        if(cv::waitKey(30)>=0){
            // Stop the stream.
            try {
                  dac.stopStream();
            }
            catch ( RtAudioError &error ) {
                  error.printMessage();
            }
            break;
        }
        swap(prevgray, gray);
    }
    return 0;
}