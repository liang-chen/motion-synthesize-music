
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/video.hpp"
#include "opencv2/highgui.hpp"

#include "RtAudio.h"

#include <iostream>
#include <math.h>
#include <unordered_map>

using namespace std;

#define FRAME_WIDTH 640
#define FRAME_HEIGHT 480
#define MIN_LEFT_RIGHT_DIS 50
#define SEARCH_RANGE 20
#define ROI_RADIOUS 20

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

static long long stateToIndex(TRACK_STATE state){
      long long index = 0;
      long long temp;
      cv::Point left = state.left;
      cv::Point rite = state.rite;
      
      index += left.x;
      index += left.y * FRAME_WIDTH;
      
      temp = rite.x * FRAME_WIDTH;
      index += temp * FRAME_HEIGHT;
      
      temp = rite.y * FRAME_WIDTH;
      index += temp * FRAME_HEIGHT * FRAME_WIDTH;
      
      return index;
}

static void indexToState(long long index, TRACK_STATE& state){
      state.rite.y = index / FRAME_WIDTH / FRAME_HEIGHT / FRAME_WIDTH;
      
      long long temp = state.rite.y*FRAME_WIDTH;
      temp *= FRAME_HEIGHT;
      temp *= FRAME_WIDTH;
      state.rite.x = (index - temp)/ FRAME_WIDTH/ FRAME_HEIGHT;
      
      temp += state.rite.x*FRAME_WIDTH*FRAME_HEIGHT;
      state.left.y = (index - temp)/FRAME_WIDTH;
      
      temp += state.left.y*FRAME_WIDTH;
      state.left.x = index - temp;
}

static void TimeContinuousTracking(const cv::Mat& flow, TRACK_STATE& obj_state, const cv::Mat& centers){

      cv::Point nominal_left = (centers.at<cv::Point2f>(0).x < centers.at<cv::Point2f>(1).x)?centers.at<cv::Point2f>(0):centers.at<cv::Point2f>(1);
      cv::Point nominal_rite = (centers.at<cv::Point2f>(0).x < centers.at<cv::Point2f>(1).x)?centers.at<cv::Point2f>(1):centers.at<cv::Point2f>(0);
      
      float opt_log_prob_left = -HUGE_VAL;
      float opt_log_prob_rite = -HUGE_VAL;
      cv::Point opt_left, opt_rite;
      
      for(int y = obj_state.left.y - SEARCH_RANGE; y < obj_state.left.y + SEARCH_RANGE; y++){
            for(int x = obj_state.left.x -SEARCH_RANGE; x < obj_state.left.x + SEARCH_RANGE; x++){
                  if(x < 0 || y < 0 || x >= flow.cols || y >= flow.rows) continue;
                  
                  cv::Point left = cv::Point(x,y);
                  float temp = -powf(cv::norm(left - nominal_left)/100, 2.0);
                  if(temp > opt_log_prob_left){
                        opt_log_prob_left = temp;
                        opt_left = left;
                  }
            
            }
      }
      
      for(int y = obj_state.rite.y - SEARCH_RANGE; y < obj_state.rite.y + SEARCH_RANGE; y++){
            for(int x = obj_state.rite.x -SEARCH_RANGE; x < obj_state.rite.x + SEARCH_RANGE; x++){
                  if(x < 0 || y < 0 || x >= flow.cols || y >= flow.rows) continue;
                  
                  cv::Point rite = cv::Point(x,y);
                  float temp = -powf(cv::norm(rite - nominal_rite)/100, 2.0);
                  if(temp > opt_log_prob_rite){
                        opt_log_prob_rite = temp;
                        opt_rite = rite;
                  }
            }
      }

      const float log_prior = logf( (0.8*expf(obj_state.log_prob) + 0.2) / (0.2*expf(obj_state.log_prob) + 0.8));
//      if(opt_log_prob_left + opt_log_prob_rite + log_prior > obj_state.log_prob){
//            obj_state.log_prob  = opt_log_prob_left + opt_log_prob_rite + log_prior;
//            obj_state.left = opt_left;
//            obj_state.rite = opt_rite;
//      }
      
      obj_state.log_prob  = opt_log_prob_left + opt_log_prob_rite + log_prior;
      obj_state.left = opt_left;
      obj_state.rite = opt_rite;
      
      cout<<"left "<<obj_state.left.x<<" "<<obj_state.left.y<<endl;
      cout<<"rite "<<obj_state.rite.x<<" "<<obj_state.rite.y<<endl;
      
//      for(int i = 0; i < candidates.size(); i++)
//      {
//            TRACK_STATE state = candidates.at(i);
//            cv::Point left = state.left;
//            cv::Point rite = state.rite;
//            
//            long long index = stateToIndex(state);
//            unordered_map<long long,float>::const_iterator got = scores.find(index);
//            if ( got == scores.end() )
//                  scores.insert({index , 0});
//            
//            float log_prob = 0;
//
//            float prev_log_prob = scores.at(index);
//            
//            const float log_prior = logf( (0.8*expf(prev_log_prob) + 0.2) / (0.2*expf(prev_log_prob) + 0.8) );
//            
//            log_prob += (thresh-(cv::norm(left - cv::Point(nominal_left.x, nominal_left.y)) + cv::norm(rite - cv::Point(nominal_rite.x, nominal_rite.y)))/50);
//            log_prob += log_prior;
//            cout << log_prob << endl;
//            scores.at(index) = log_prob;
//      }
//      
//      for(auto i: scores){
//            if( i.second > opt){
//                  opt = i.second;
//                  indexToState(i.first, opt_state);
//                  opt_state.log_prob = opt;
//            }
//      }
//      
//      if(opt_state.log_prob > 0)
//            obj_state = opt_state;

}

static void findCenters(vector<cv::Point2f> salient_points, vector<int> labels, vector<float> weights, cv::Mat& centers){
      cv::Point2f left;
      cv::Point2f rite;
      float cnt_left, cnt_rite;
      

      int iter = 0;
      while(iter < 10){
            left.x = left.y = rite.x = rite.y = cnt_left = cnt_rite = 0;
            for(int i = 0; i < salient_points.size(); i++){
                  cv::Point2f point = salient_points.at(i);
                  if(labels.at(i) == 0)
                  {
                        left.x += point.x * weights.at(i);
                        left.y += point.y * weights.at(i);
                        cnt_left += weights.at(i);
                  }
                  else{
                        rite.x += point.x * weights.at(i);
                        rite.y += point.y * weights.at(i);
                        cnt_rite += weights.at(i);
                  }
            }
            
            if(cnt_left == 0) left = salient_points.at(0);
            else{
                  left.x /= cnt_left;
                  left.y /= cnt_left;
            }
            
            if(cnt_rite == 0) rite = salient_points.at(0);
            else{
                  rite.x /= cnt_rite;
                  rite.y /= cnt_rite;
            }
            
            for(int i = 0; i < labels.size(); i++){
                  labels.at(i) = (cv::norm(salient_points.at(i) - left) > cv::norm(salient_points.at(i) - rite))?1:0;
            }
            iter++;
            
      }
      
      centers.at<cv::Point2f>(0) = left;
      centers.at<cv::Point2f>(1) = rite;
      
}

static void TrackHands(const cv::Mat& flow, unordered_map<long long, float>& scores, TRACK_STATE& obj_state){

      float thresh = 10;
      cv::Point left = obj_state.left;
      cv::Point rite = obj_state.rite;
      vector<cv::Point2f> salient_points;
      vector<int> labels;
      vector<float> weights;
      
      for(int y = 0; y < flow.rows; y++){
            for(int x = 0; x < flow.cols; x++){
                  if(cv::norm(flow.at<cv::Point2f>(y,x)) > thresh){
                        salient_points.push_back(cv::Point2f(x,y));
                        weights.push_back(cv::norm(flow.at<cv::Point2f>(y,x))/400.0);
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
      cv::Mat centers(2, 1, CV_32FC2);
      
      //cv::kmeans(mPoints, 2, mLabels,
      //      cv::TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 3, 0.001),
      //         3, cv::KMEANS_PP_CENTERS, centers);
      
      
      findCenters(salient_points, labels, weights, centers);
      
//      obj_state.left = (centers.at<cv::Point2f>(0).x < centers.at<cv::Point2f>(1).x) ? centers.at<cv::Point2f>(0):centers.at<cv::Point2f>(1);
//      obj_state.rite = (centers.at<cv::Point2f>(0).x < centers.at<cv::Point2f>(1).x) ? centers.at<cv::Point2f>(1):centers.at<cv::Point2f>(0);
//      obj_state.log_prob = 1;
//      return;
      
//      sort(salient_points.begin(), salient_points.end(),
//            [](const cv::Point2f & a, const cv::Point2f & b) -> bool
//            {
//                  return a.x > b.x;
//            });
//      
//      if(salient_points.size() > 1000)
//            salient_points.resize(1000);
//      
//      cv::Mat uflow(flow.rows, flow.cols, CV_32F);
//      for(int i = 0; i < uflow.rows; i++){
//            for(int j = 0; j < uflow.cols; j++){
//                  uflow.at<float>(i, j) = cv::norm(flow.at<cv::Point2f>(i,j));
//            }
//      }
//      
//      vector<TRACK_STATE> states;
//      for(int i = 0; i < salient_points.size(); i++){
//            for(int j = i + 1; j < salient_points.size(); j++){
//                  if(fabs(salient_points.at(j).x - salient_points.at(i).x) > MIN_LEFT_RIGHT_DIS){
//                        TRACK_STATE state;
//                        state.left = salient_points.at(i);
//                        state.rite = salient_points.at(j);
//                        state.log_prob = uflow.at<float>(state.left.y, state.left.x) + uflow.at<float>(state.rite.y, state.rite.x);
//                        states.push_back(state);
//                  }
//            }
//      }
//      
//      sort(states.begin(), states.end(),
//            [](const TRACK_STATE & a, const TRACK_STATE & b) -> bool
//            {
//                  return a.log_prob > b.log_prob;
//            });
//      if(states.size() > 200)
//            states.resize(200);
      
      TimeContinuousTracking(flow, obj_state, centers);
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
    
    long long index = stateToIndex(obj_state);
    cout<< "first:"<<obj_state.left.x <<" "<< obj_state.left.y<<" "<< obj_state.rite.x<<" "<< obj_state.rite.y<< endl;
    
    indexToState(index, obj_state);
    
    cout<< "first:"<<obj_state.left.x <<" "<< obj_state.left.y<<" "<< obj_state.rite.x<<" "<< obj_state.rite.y<< endl;
    //int size[4] = { FRAME_WIDTH, FRAME_HEIGHT, FRAME_WIDTH, FRAME_HEIGHT };
    //cv::Mat scores(4, size, CV_32F, cv::Scalar(0));
    
    unordered_map <long long, float> scores;
    for(;;)
    {
        cap >> frame;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        if( !prevgray.empty() )
        {
            cv::calcOpticalFlowFarneback(prevgray, gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
            cv::cvtColor(prevgray, cflow, cv::COLOR_GRAY2BGR);
            TrackHands(flow, scores, obj_state);
//            if(obj_state.log_prob > 0){
//                  //mag_global = cv::norm(obj_state.left.x - obj_state.rite.x)/50;
//                  //frequency_global = 440*(0.5 + cv::norm(obj_state.left.y - obj_state.rite.y)/80);
//            }
//            else{
//                  //InitTrackState(obj_state, FRAME_HEIGHT, FRAME_WIDTH);
//            }
            mag_global = cv::norm(obj_state.left.x - obj_state.rite.x)/100;
            cout << mag_global << endl;
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