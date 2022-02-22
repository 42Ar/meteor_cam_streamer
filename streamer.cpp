#include <iostream>
#include "json.hpp"
#include <string>
#include <fstream>
#include <cmath>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/ocl.hpp>
#include <limits>
#include <sstream>
#include <iomanip>

using namespace std;
using namespace nlohmann;
using namespace cv;

typedef Mat_<double> Matd;

const static string main_config_file = "config.json";
vector<int> active_cameras;
int in_size_x, in_size_y;
int out_size_x, out_size_y;
bool test_mode, start_stream, use_test_images, verbose_mainloop;
string rtmp_url, test_output_file;
int fps;
int bitrate;
Mat mask;
double vignette_correction;
bool enable_vignette_correction;
VideoWriter video;
int selected_device;

const int OUTSIDE_ROI = -1;

struct camera{
    Mat src, map_x, map_y, cur_img;
    int id;
    vector<double> c;
    double fov_x, fov_y;
    double az, alt, roll;
    double crop_az[2];
    double crop_alt[2];
    Point upper_left, lower_right;
    Matx33d M;
    json calib;
    string url, test_img;
    VideoCapture cap;

    camera(int id):id(id){}
};
std::vector<camera> cams;

int rpoly(double *op, int degree, double *zeror, double *zeroi);

Matx33d Rx(double angle){
    return Matx33d(1,           0,          0,
                   0,  cos(angle), sin(angle),
                   0, -sin(angle), cos(angle));
}

Matx33d Rz(double angle){
    return Matx33d( cos(angle), sin(angle), 0,
                   -sin(angle), cos(angle), 0,
                             0,          0, 1); 
}

void ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
}

void rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), s.end());
}

void trim(std::string &s) {
    ltrim(s);
    rtrim(s);
}

void read_config(const string &config_file){
    cout << "opening main config file " << config_file << endl;
    ifstream file(config_file);
    if(!file.is_open()){
        cerr << "failed to open main config file " << main_config_file << endl;
        exit(1);
    }
    json config;
    file >> config;
    file.close();
    in_size_x = config["in_size_x"];
    in_size_y = config["in_size_y"];
    out_size_x = config["out_size_x"];
    out_size_y = config["out_size_y"];
    use_test_images = config["use_test_images"];
    verbose_mainloop = config["verbose_mainloop"];
    selected_device = config["selected_device"];
    test_mode = config["test_mode"];
    if(test_mode){
        test_output_file = config["test_output_file"];
    }
    start_stream = config["start_stream"];
    fps = config["fps"];
    vignette_correction = config["vignette_correction"];
    enable_vignette_correction = config["enable_vignette_correction"];
    if(start_stream){
        string rtmp_url_file = config["rtmp_url_file"];
        ifstream f(rtmp_url_file);
        if(!f.is_open()){
            cerr << "failed to open file containing rtmp url: " << rtmp_url_file << endl;
            exit(1);
        }
        stringstream buffer;
        buffer << f.rdbuf();
        f.close();
        rtmp_url = buffer.str();
        trim(rtmp_url);
    }
    bitrate = config["bitrate"];
    for(auto id : config["active_cams"]){
        cams.push_back(camera(id));
    }
    for(auto &calib_file : config["calib_input_files"]){
        ifstream cf(calib_file.get<string>());
        cout << "reading calib data from " << calib_file.get<string>() << endl;
        if(!cf.is_open()){
            cerr << "failed to calibration data file " << calib_file << endl;
            exit(1);
        }
        json calib_data;
        cf >> calib_data;
        cf.close();
        for(auto cam_calib : calib_data){
            int cam_calib_id = cam_calib["id"];
            for(auto &cam : cams){
                if(cam.id == cam_calib_id){
                    cam.calib = cam_calib;
                    break;
                }
            }
        }
    }
    for(auto &cam : cams){
        if(cam.calib.is_null()){
            cerr << "no calibration data for camera " << cam.id << endl;
            exit(1);
        }
        cam.fov_x = cam.calib["fov"][0];
        cam.fov_y = cam.calib["fov"][1];
        cam.az = cam.calib["az"];
        cam.alt = cam.calib["alt"];
        cam.roll = cam.calib["roll"];
        int order = 2;
        while(cam.calib.contains("c_" + to_string(order))){
            cam.c.push_back(cam.calib["c_" + to_string(order)]);
            order += 1;
        }
        json cam_inf = config["cameras"][to_string(cam.id)];
        if(cam_inf.is_null()){
            cerr << "please add configuration for camera " << cam.id << endl;
            exit(1);
        }
        if(use_test_images){
            cam.test_img = cam_inf["test_img"];
        }else{
            cam.url = cam_inf["url"];
        }
        for(int i = 0; i < 2; i++){
            cam.crop_az[i] = cam_inf["crop_az"][i];
            cam.crop_alt[i] = cam_inf["crop_alt"][i];
        }
        if(cam.crop_az[1] < 0){
            cam.crop_az[1] = 2*M_PI - cam.crop_az[1];
        }
        if(cam.crop_az[0] < 0 || cam.crop_az[0] > 2*M_PI){
            cerr << "crop_az[0] of camera " << cam.id << " out of range" << endl;
            exit(1);
        }
        if(cam.crop_az[1] < 0 || cam.crop_az[1] > 4*M_PI){
            cerr << "crop_az[1] of camera " << cam.id << " out of range" << endl;
            exit(1);
        }
        if(cam.crop_az[1] <= cam.crop_az[0]){
            cerr << "crop_az[0] <= crop_az[1] for camera " << cam.id << endl;
            exit(1);
        }
        for(int i = 0; i < 2; i++){
            if(cam.crop_alt[i] < -M_PI_2 || cam.crop_alt[i] > M_PI_2){
                cerr << "crop_alt[" << i << "] of camera " << cam.id << " out of range" << endl;
                exit(1);
            }
        }
        if(cam.crop_alt[0] >= cam.crop_alt[1]){
            cerr << "crop_alt[0] >= crop_alt[1] for camera " << cam.id << endl;
            exit(1);
        }
    }
    cout << "active cameras:";
    for(auto &cam : cams){
        cout << " " << cam.id;
    }
    cout << endl;
}

Vec3d pixel_to_vec(const camera &cam, int px, int py){
    double x = double(2*px)/(in_size_x - 1) - 1;
    double y = double(2*py)/(in_size_y - 1) - 1;
    y = y*cam.fov_y/cam.fov_x;
    if(cam.c.size() >= 2){
        double r = sqrt(x*x + y*y);
        double r_corr = r;
        for(int i = 0; i < cam.c.size(); i++){
            r_corr += cam.c[i]*pow(r, i + 2); 
        }
        x = x*r_corr/r;
        y = y*r_corr/r;
    }
    return cam.M*Vec3d(x*cam.fov_x, y*cam.fov_x, 1);
}

Vec2d vec_to_pixel(const camera &cam, const Vec3d &vec){
    Vec3d v = cam.M.t()*vec;
    if(v[2] <= 0){
        return Vec2d(OUTSIDE_ROI, OUTSIDE_ROI);
    }
    double x = v[0]/(cam.fov_x*v[2]);
    double y = v[1]/(cam.fov_x*v[2]);
    if(cam.c.size() >= 2){
        double r_corr = sqrt(x*x + y*y);
        vector<double> op(cam.c.size() + 2);
        op[cam.c.size() + 1] = -r_corr;
        op[cam.c.size()] = 1;
        for(int i = 0; i < cam.c.size(); i++){
            op[cam.c.size() - 1 - i] = cam.c[i];
        }
        vector<double> R(op.size() - 1), I(op.size() - 1);
        int ret = rpoly(op.data(), op.size() - 1, R.data(), I.data());
        if(ret == -1){
            cerr << "Error finding roots. Is the leading correction coeffcient zero?" << endl;
            exit(1);
        }
        float r = -1;
        for(int i = 0; i < ret; i++){
            if(abs(I[i]) < numeric_limits<double>::epsilon() && R[i] > 0){
                r = R[i];
                break;
            }
        }
        if(r == -1){
            cerr << "No suitable root found. Are the correction coefficients correct?" << endl;
            exit(1);
        }
        x = x*r/r_corr;
        y = y*r/r_corr;
    }
    y = y*cam.fov_x/cam.fov_y;
    double px = (in_size_x - 1)*(x + 1)/2;
    double py = (in_size_y - 1)*(y + 1)/2;
    return Vec2d(px, py);
}

Vec2d inverse_project_equirect(int px, int py){
    double az = 2*M_PI*double(px)/out_size_x;
    double alt = -M_PI*(double(py)/(out_size_y - 1) - 0.5);
    return Vec2d(az, alt);

}

Vec2d project_equirect(Vec2d az_alt){
    return Vec2d(az_alt[0]*out_size_x/(2*M_PI), (0.5 - az_alt[1]/M_PI)*(out_size_y - 1));
}

Vec3d spherical_to_cartesian(Vec2d az_alt){
    return Vec3d(sin(az_alt[0])*cos(az_alt[1]),
                 cos(az_alt[0])*cos(az_alt[1]),
                 sin(az_alt[1]));
}

void precalc_pixel_grids(){
    cout << "pre calculating grid for camera:" << flush;
    for(auto &cam : cams){
        cout << " " << cam.id << flush;
        cam.M = Rz(cam.az)*Rx(M_PI_2 - cam.alt)*Rz(cam.roll);
        int test_x = 55, test_y = 21;
        Vec3d test_v = pixel_to_vec(cams[0], test_x, test_y);
        Vec2d test_ret = vec_to_pixel(cams[0], test_v);
        if(abs(test_ret[0] - test_x) > 0.0001 || abs(test_ret[1] - test_y) > 0.0001){
            cerr << "transform testing failed" << endl;
            exit(1);
        }
        Vec2d upper_left = project_equirect(Vec2d(cam.crop_az[0], cam.crop_alt[1]));
        Vec2d lower_right = project_equirect(Vec2d(cam.crop_az[1], cam.crop_alt[0]));
        cam.upper_left.x = int(ceil(upper_left[0]));
        cam.upper_left.y = int(ceil(upper_left[1]));
        cam.lower_right.x = int(floor(lower_right[0]));
        cam.lower_right.y = int(floor(lower_right[1]));
        Size size = Size(cam.lower_right - cam.upper_left + Point(1, 1));
        cam.map_x.create(size, CV_32FC1);
        cam.map_y.create(size, CV_32FC1);
        for(int y = 0; y < size.height; y++){
            for(int x = 0; x < size.width; x++){
                Vec2d s = inverse_project_equirect(cam.upper_left.x + x, cam.upper_left.y + y);
                Vec3d v = spherical_to_cartesian(s);
                Vec2d p = vec_to_pixel(cam, v);
                if(p[0] < 0 || p[1] < 0 || p[0] > in_size_x - 1 || p[1] > in_size_y - 1){
                    //pixel outside ROI
                }
                cam.map_x.at<float>(y, x) = p[0];
                cam.map_y.at<float>(y, x) = p[1];
            }
        }
    }
    cout << endl;
}

void precalc_brightness_mask(){
    cout << "pre calculating brightness mask" << endl;
    mask.create(in_size_y, in_size_x, CV_32FC1);
    for(int y = 0; y < in_size_y; y++){
        for(int x = 0; x < in_size_x; x++){
            float xd = (x - (in_size_x - 1)/2.0)/((in_size_x - 1)/2.0);
            float yd = (y - (in_size_y - 1)/2.0)/((in_size_x - 1)/2.0);
            float r = sqrt(xd*xd + yd*yd);
            mask.at<float>(y, x) = 1/cos(atan(r/vignette_correction));
        }
    }
}


void process(Mat &dst){
    for(auto &cam : cams){
        if(enable_vignette_correction){
            for(int y = 0; y < in_size_y; y++){
                for(int x = 0; x < in_size_x; x++){
                    cam.cur_img.at<Vec3b>(y, x) *= mask.at<float>(y, x);
                }
            }
        }
        if(cam.lower_right.x >= out_size_x){
            Mat roi_right(dst, Rect(cam.upper_left, Point(out_size_x, cam.lower_right.y + 1)));
            Mat roi_right_map_x(cam.map_x, Rect(0, 0, out_size_x - cam.upper_left.x, cam.map_x.size().height));
            Mat roi_right_map_y(cam.map_y, Rect(0, 0, out_size_x - cam.upper_left.x, cam.map_y.size().height));
            remap(cam.cur_img, roi_right, roi_right_map_x, roi_right_map_y, INTER_LINEAR, BORDER_TRANSPARENT);

            Mat roi_left(dst, Rect(Point(0, cam.upper_left.y), Point(cam.lower_right.x - out_size_x + 1, cam.lower_right.y + 1)));
            Mat roi_left_map_x(cam.map_x, Rect(Point(out_size_x - cam.upper_left.x, 0), Point(cam.map_x.size())));
            Mat roi_left_map_y(cam.map_y, Rect(Point(out_size_x - cam.upper_left.x, 0), Point(cam.map_y.size())));
            remap(cam.cur_img, roi_left, roi_left_map_x, roi_left_map_y, INTER_LINEAR, BORDER_TRANSPARENT);
        }else{
            Mat roi(dst, Rect(cam.upper_left, cam.lower_right + Point(1, 1)));
            remap(cam.cur_img, roi, cam.map_x, cam.map_y, INTER_LINEAR, BORDER_TRANSPARENT);
        }
    }
}

void open_cameras(){
    cout << "opening camera:" << flush;
    for(auto &cam : cams){
        cout << " " << cam.id << flush;
        if(!cam.cap.open(cam.url)){
            cerr << "failed to open camera " << cam.id << endl;
            exit(1);
        }
    }
    cout << endl;
}

void read_frames(){
    if(!use_test_images){
        for(auto &cam : cams){
            cam.cap.grab();
        }
    }
    for(auto &cam : cams){
        if(use_test_images){
            cam.cur_img = imread(cam.test_img, IMREAD_COLOR);
            if(cam.cur_img.empty()){
                cerr << "failed to read test image " << cam.test_img << endl;
                exit(1);
            }
        }else{
            cam.cap.retrieve(cam.cur_img);
        }
    }
}

void open_pipeline(){
    cout << "opening pipline" << endl;
    stringstream pipeline;
    pipeline << "appsrc ! ";
    pipeline << "video/x-raw, format=BGR, width=" << out_size_x << ", height=" << out_size_y << ", framerate=" << fps << "/1 ! ";
    pipeline << "queue ! ";
    pipeline << "videoconvert ! ";
    pipeline << "x264enc bitrate=" << bitrate << " byte-stream=false key-int-max=" << fps*2 << " bframes=0 aud=true ! "; // key frames must appear every two seconds at max
    pipeline << "video/x-h264,profile=main ! ";
    pipeline << "flvmux streamable=true name=mux ! ";
    pipeline << "rtmpsink location=\"" << rtmp_url << "\" " << endl;
    pipeline << "audiotestsrc wave=4 ! voaacenc bitrate=0 ! mux. " << endl; // add silent dummy audio
    video.open(pipeline.str(), CAP_GSTREAMER, 0, fps, Size(out_size_x, out_size_y), true); 
}

void setup_opencl(){
    ocl::setUseOpenCL(true);
    if(!ocl::haveOpenCL()){
        cout << "OpenCL is not available" << endl;
        return;
    }
    cv::ocl::Context context;
    if(!context.create(ocl::Device::TYPE_GPU)){
        cout << "failed creating an opencl GPU context" << endl;
        return;
    }
    cout << context.ndevices() << " GPU devices are detected." << endl;
    for(int i = 0; i < context.ndevices(); i++){
        ocl::Device device = context.device(i);
        cout << "=== Device " << i << endl;
        cout << "name:              " << device.name() << endl;
        cout << "available:         " << device.available() << endl;
        cout << "imageSupport:      " << device.imageSupport() << endl;
        cout << "OpenCL_C_Version:  " << device.OpenCL_C_Version() << endl;
        cout << endl;
    }
    if(selected_device < context.ndevices()){
        ocl::Device(context.device(selected_device));
    }else{
        cout << "selected device index out of range" << endl;
    }
}


int main(int argc, char *argv[]){
    string config = main_config_file;
    if(argc == 2){
        config = argv[1];
    }
    read_config(config);
    setup_opencl();
    if(enable_vignette_correction){
        precalc_brightness_mask();
    }
    precalc_pixel_grids();
    if(!use_test_images){
        open_cameras();
    }
    Mat dst(out_size_y, out_size_x, CV_8UC3, Scalar(0, 0, 0));
    if(start_stream){
        open_pipeline();
    }
    cout << "starting mainloop" << endl;
    double tick_freq = getTickFrequency();
    while(true){
        int64 start_frame = getTickCount();
        if(verbose_mainloop){
            cout << "reading frames" << endl;
        }
        int64 read_frames_start = getTickCount();
        read_frames();
        int64 read_frames_end = getTickCount();
        if(verbose_mainloop){
            cout << "processing frames" << endl;
        }
        int64 process_frames_start = getTickCount();
        process(dst);
        int64 process_frames_end = getTickCount();
        if(test_mode){
            imwrite(test_output_file, dst);
            imshow("output", dst);
            while(waitKey() != 'q');
            destroyAllWindows();
        }
        int64 write_frames_start = 0, write_frames_end = 0;
        if(start_stream){
            if(verbose_mainloop){
                cout << "writing frame" << endl;
            }
            write_frames_start = getTickCount();
            video << dst;
            write_frames_end = getTickCount();
        }
        int64 end_frame = getTickCount();
        double fps = tick_freq/(end_frame - start_frame);
        cout << setiosflags(ios::fixed) << setprecision(1);
        cout << "read:" << setw(6) << 1e3*(read_frames_end - read_frames_start)/tick_freq << " ms, "
             << "process:" << setw(6) << 1e3*(process_frames_end - process_frames_start)/tick_freq << " ms, "
             << "write:" << setw(6) << 1e3*(write_frames_end - write_frames_start)/tick_freq << " ms, "
             << "fps:" << setw(6) << fps << endl;
    }
    return 0;
}
