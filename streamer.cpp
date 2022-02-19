#include <iostream>
#include "json.hpp"
#include <string>
#include <fstream>
#include <cmath>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <limits>
#include <sstream>

using namespace std;
using namespace nlohmann;
using namespace cv;

typedef Mat_<double> Matd;

const static string main_config_file = "config.json";
vector<int> active_cameras;
int in_size_x, in_size_y;
int out_size_x, out_size_y;
bool test_mode;
vector<string> test_images({
    "../starfitter/fit_images/2022_02_13_20_10_01_1.jpg",
    "../starfitter/fit_images/2022_02_13_20_10_01_2.jpg",
    "../starfitter/fit_images/2022_02_13_20_10_00_3.jpg",
    "../starfitter/fit_images/2022_02_13_20_10_00_4.jpg",
    "../starfitter/fit_images/2022_02_13_20_10_00_5.jpg",
    "../starfitter/fit_images/2022_02_13_20_10_00_6.jpg",
    "../starfitter/fit_images/2022_02_13_20_10_02_7.jpg"
});
string rtmp_url;
int fps;
int bitrate;

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

void read_config(){
    ifstream file(main_config_file);
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
    test_mode = config["test_mode"];
    fps = config["fps"];
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
        json crop = config["crop"][to_string(cam.id)];
        if(crop.is_null()){
            cerr << "please add crop info form camera " << cam.id << endl;
            exit(1);
        }
        for(int i = 0; i < 2; i++){
            cam.crop_az[i] = crop["az"][i];
            cam.crop_alt[i] = crop["alt"][i];
        }
        if(cam.crop_az[1] < 0){
            cam.crop_az[1] = 2*M_PI - cam.crop_az[1];
        }
        if(cam.crop_az[0] < 0 || cam.crop_az[0] > 2*M_PI){
            cerr << "az crop[0] of camera " << cam.id << " out of range" << endl;
            exit(1);
        }
        if(cam.crop_az[1] < 0 || cam.crop_az[1] > 4*M_PI){
            cerr << "az crop[1] of camera " << cam.id << " out of range" << endl;
            exit(1);
        }
        if(cam.crop_az[1] <= cam.crop_az[0]){
            cerr << "az crop[0] <= crop[1] for camera " << cam.id << endl;
            exit(1);
        }
        for(int i = 0; i < 2; i++){
            if(cam.crop_alt[i] < -M_PI_2 || cam.crop_alt[i] > M_PI_2){
                cerr << "alt crop[" << i << "] of camera " << cam.id << " out of range" << endl;
                exit(1);
            }
        }
        if(cam.crop_alt[0] >= cam.crop_alt[1]){
            cerr << "alt crop[0] >= crop[1] for camera " << cam.id << endl;
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
    for(auto &cam : cams){
        cout << "pre calculating grid for camera " << cam.id << endl;
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
        for(int x = 0; x < size.width; x++){
            for(int y = 0; y < size.height; y++){
                Vec2d s = inverse_project_equirect(cam.upper_left.x + x, cam.upper_left.y + y);
                Vec3d v = spherical_to_cartesian(s);
                Vec2d p = vec_to_pixel(cam, v);
                if(p[0] < 0 || p[1] < 0 || p[0] > in_size_x - 1 || p[1] > in_size_y - 1){
                    //pixel outside ROI
                }
                cam.map_x.at<float>(y, x%out_size_x) = p[0];
                cam.map_y.at<float>(y, x%out_size_x) = p[1];
            }
        }
    }
}


void process(Mat &dst){
    for(auto &cam : cams){
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


int main(){
    read_config();
    Mat dst(out_size_y, out_size_x, CV_8UC3, Scalar(0, 0, 0));
    if(test_mode){
        for(auto &cam : cams){
            cam.cur_img = imread(test_images[cam.id - 1], IMREAD_COLOR);
        }
    }
    precalc_pixel_grids();
    process(dst);
    if(test_mode){
        imwrite("output.jpg", dst);
    }
    stringstream pipeline;
    pipeline << "appsrc ! ";
    pipeline << "video/x-raw, format=BGR, width=" << out_size_x << ", height=" << out_size_y << ", framerate=" << fps << "/1 ! ";
    pipeline << "queue ! ";
    pipeline << "videoconvert ! ";
    pipeline << "x264enc bitrate=" << bitrate << " byte-stream=false key-int-max=60 bframes=0 aud=true ! ";
    pipeline << "video/x-h264,profile=main ! ";
    pipeline << "flvmux streamable=true name=mux ! ";
    pipeline << "rtmpsink location=\"" << rtmp_url << "\" " << endl;
    pipeline << "audiotestsrc ! voaacenc bitrate=128000 ! mux. " << endl;
    VideoWriter video(pipeline.str(), CAP_GSTREAMER, 0, fps, Size(out_size_x, out_size_y), true); 
    cout << "opened" << endl;
    for(int i = 0; true; i++){
        video << dst;
        cout << i << endl;
    } 
    return 0;
}