#include <iostream>
#include "json.hpp"
#include <string>
#include <fstream>
#include <cmath>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <limits>
#include <sstream>
#include <iomanip>

using namespace std;
using namespace nlohmann;
using namespace cv;

typedef Mat_<double> Matd;

const static string main_config_file = "config.json";
int in_size_x, in_size_y;
int out_size_x, out_size_y;
bool test_mode, start_stream, use_test_images, verbose_mainloop;
string rtmp_url, test_output_file;
int fps;
int bitrate;
double mask_scale;
double alt_start, alt_end;
double vignette_correction;
double az_offset;
bool enable_vignette_correction, enable_transparency, calculate_coverage;
VideoWriter video;
int selected_device;

const int OUTSIDE_ROI = -1;
const float pixel_eps = 1e-8;

struct camera{
    cuda::GpuMat src, map_x, map_y, cur_img, dst_roi;
    Mat decoded_img;
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
struct cam_batch{
    vector<int> cams;
    cuda::GpuMat mask;
};
vector<cam_batch> cam_batches;
vector<camera> cams;

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
    if(config["in_size_x"].is_null() || config["in_size_y"].is_null()){
        cerr << "please specify in_size_x and in_size_y" << endl;
        exit(1);
    }
    in_size_x = config["in_size_x"];
    in_size_y = config["in_size_y"];
    if(config["out_size_x"].is_null() || config["out_size_y"].is_null()){
        cerr << "please specify out_size_x and out_size_y" << endl;
        exit(1);
    }
    out_size_x = config["out_size_x"];
    out_size_y = config["out_size_y"];
    calculate_coverage = config.value("calculate_coverage", false);
    az_offset = config.value("az_offset", 0);
    use_test_images = config.value("use_test_images", false);
    verbose_mainloop = config.value("verbose_mainloop", false);
    selected_device = config.value("selected_device", 0);
    test_mode = config.value("test_mode", false);
    if(test_mode){
        test_output_file = config["test_output_file"];
    }
    alt_start = config.value("alt_start", -M_PI_2);
    alt_end = config.value("alt_end", M_PI_2);
    start_stream = config.value("start_stream", false);
    enable_vignette_correction = config.value("enable_vignette_correction", false);
    if(enable_vignette_correction){
        vignette_correction = config["vignette_correction"];
    }
    if(start_stream){
        if(config["fps"].is_null()){
            cerr << "please specify the fps" << endl;
            exit(1);
        }
        fps = config["fps"];
        if(config["rtmp_url_file"].is_null()){
            cerr << "please specify the rtmp_url_file" << endl;
            exit(1);
        }
        string rtmp_url_file = config["rtmp_url_file"];
        if(config["bitrate"].is_null()){
            cerr << "please specify the bitrate" << endl;
            exit(1);
        }
        bitrate = config["bitrate"];
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
    cam_batches.resize(config["active_cams"].size());
    enable_transparency = config.value("enable_transparency", false);
    if(enable_transparency && cam_batches.size() == 1){
        cout << "enabeling transparency makes no sense if there is only one batch" << endl;
        enable_transparency = false;
    }
    for(int i = 0; i < cam_batches.size(); i++){
        for(auto &id : config["active_cams"][i]){
            cams.push_back(camera(id));
            cam_batches[i].cams.push_back(cams.size() - 1);
        }
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
        for(auto &cam_calib : calib_data){
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
        if(cam.calib["fov"][0].is_null() || cam.calib["fov"][1].is_null() ||
           cam.calib["az"].is_null() || cam.calib["alt"].is_null() || cam.calib["roll"].is_null()){
            cerr << "cam calibration data malformed" << endl;
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
            if(cam_inf["test_img"].is_null()){
                cerr << "please specify test_img as a path to the image file when using test images" << endl;
                exit(1);
            }
            cam.test_img = cam_inf["test_img"];
        }else{
            if(cam_inf["url"].is_null()){
                cerr << "please specify url to the camera when start_stream is true" << endl;
                exit(1);
            }
            cam.url = cam_inf["url"];
        }
        for(int i = 0; i < 2; i++){
            if(cam_inf["crop_az"][i].is_null()){
                cerr << "please specify crop_az as an array of 2 values" << endl;
                exit(1);
            }
            cam.crop_az[i] = cam_inf["crop_az"][i];
            if(cam_inf["crop_alt"][i].is_null()){
                cerr << "please specify crop_alt as an array of 2 values" << endl;
                exit(1);
            }
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
    double az_norm = (double(px) + 0.5)/out_size_x;
    double alt_norm = double(py + 0.5)/out_size_y;
    double alt = alt_end*(1.0 - alt_norm) + alt_start*alt_norm;
    return Vec2d(2*M_PI*az_norm - az_offset, alt);

}

Vec2d project_equirect(Vec2d az_alt){
    double alt_norm = (alt_end - az_alt[1])/(alt_end - alt_start);
    double az_norm = (az_alt[0] + az_offset)/(2*M_PI);
    return Vec2d(az_norm*out_size_x - 0.5, alt_norm*out_size_y - 0.5);
}

Vec3d spherical_to_cartesian(Vec2d az_alt){
    return Vec3d(sin(az_alt[0])*cos(az_alt[1]),
                 cos(az_alt[0])*cos(az_alt[1]),
                 sin(az_alt[1]));
}

int to_pixel(float i, int size){
    int p = int(round(i));
    if(p < 0){
        if(i >= -0.5 - pixel_eps){
            return 0;
        }
        cerr << "pixel coord < -0.5 - pixel_eps" << endl;
        exit(1);
    }
    if(p > size){
        if(i <= size - 0.5 + pixel_eps){
            return size - 1;
        }
        cerr << "pixel coord > size - 0.5 - pixel_eps" << endl;
        exit(1);
    }
    return p;
}

Vec2i to_pixel(Vec2d p, Vec2i size){
    return Vec2i(to_pixel(p[0], size[0]),
                 to_pixel(p[1], size[1]));
}

Vec2d wrap_az_coord(Vec2d v){
    v[0] -= v[0]/(2*M_PI);
    if(v[0] < 0){
        v[0] += 2*M_PI;
    }
    return v;
}

void precalc_pixel_grid(camera &cam, Mat mask, bool use_mask){
    cam.M = Rz(cam.az)*Rx(M_PI_2 - cam.alt)*Rz(cam.roll);
    int test_x = 55, test_y = 21;
    Vec3d test_v = pixel_to_vec(cam, test_x, test_y);
    Vec2d test_ret = vec_to_pixel(cam, test_v);
    if(abs(test_ret[0] - test_x) > 0.0001 || abs(test_ret[1] - test_y) > 0.0001){
        cerr << "transform testing failed" << endl;
        exit(1);
    }
    Vec2d upper_left = project_equirect(Vec2d(cam.crop_az[0], cam.crop_alt[1]));
    Vec2d lower_right = project_equirect(Vec2d(cam.crop_az[1], cam.crop_alt[0]));
    cam.upper_left.x = round(int(upper_left[0]));
    cam.upper_left.y = round(int(upper_left[1]));
    cam.lower_right.x = round(int(lower_right[0]));
    cam.lower_right.y = round(int(lower_right[1]));
    int shift = out_size_x*(cam.upper_left.x/out_size_x);
    cam.lower_right.x -= shift;
    cam.upper_left.x -= shift;
    Size size = Size(cam.lower_right - cam.upper_left + Point(1, 1));
    Mat map_x(size, CV_32FC1);
    Mat map_y(size, CV_32FC1);
    for(int y = 0; y < size.height; y++){
        for(int x = 0; x < size.width; x++){
            int px = cam.upper_left.x + x;
            int py = cam.upper_left.y + y;
            Vec2d s = inverse_project_equirect(px, py);
            Vec3d v = spherical_to_cartesian(s);
            Vec2d p = vec_to_pixel(cam, v);
            if(use_mask && p[0] > 0 && p[1] > 0 && p[0] < in_size_x - 1 && p[1] < in_size_y - 1){
                mask.at<unsigned char>(py, px) = static_cast<unsigned char>(255);
            }
            map_x.at<float>(y, x) = p[0];
            map_y.at<float>(y, x) = p[1];
        }
    }
    cam.map_x.create(size, CV_32FC1);
    cam.map_x.upload(map_x);
    cam.map_y.create(size, CV_32FC1);
    cam.map_y.upload(map_y);
}

void precalc_pixel_grids(){
    cout << "pre calculating grid for camera:" << flush;
    Mat mask;
    if(enable_transparency || calculate_coverage){
        mask.create(out_size_y, out_size_x, CV_8UC1);
    }
    for(int batchi = 0; batchi < cam_batches.size(); batchi++){
        cam_batch &batch = cam_batches[batchi];
        bool need_batch_mask = (enable_transparency && batchi > 0) || calculate_coverage;
        if(need_batch_mask){
            mask = static_cast<unsigned char>(0);
        }
        for(int cam : batch.cams){
            cout << " " << cams[cam].id << flush;
            precalc_pixel_grid(cams[cam], mask, need_batch_mask);
        }
        if(need_batch_mask){
            batch.mask.create(out_size_y, out_size_x, CV_8UC1);
            batch.mask.upload(mask);
        }
    }
    cout << endl;
}

double calc_brightness_correction_val(int x, int y){
    float xd = (x - (in_size_x - 1)/2.0)/((in_size_x - 1)/2.0);
    float yd = (y - (in_size_y - 1)/2.0)/((in_size_x - 1)/2.0);
    float r = sqrt(xd*xd + yd*yd);
    return 1/cos(atan(r/vignette_correction));
}

void precalc_brightness_mask(cuda::GpuMat &mask){
    cout << "pre calculating brightness mask" << endl;
    Mat m(in_size_y, in_size_x, CV_8UC3);
    double corr_max = calc_brightness_correction_val(0, 0);
    mask_scale = corr_max/255;
    for(int y = 0; y < in_size_y; y++){
        for(int x = 0; x < in_size_x; x++){
            double v = round(calc_brightness_correction_val(x, y)/mask_scale);
            m.at<Vec3b>(y, x) = Vec3b(v, v, v);
        }
    }
    mask.upload(m);
}

void process_cam(const camera &cam, cuda::GpuMat &dst, cuda::GpuMat &vignette_mask){
    if(enable_vignette_correction){
        cuda::multiply(cam.cur_img, vignette_mask, cam.cur_img, mask_scale);
    }
    if(cam.lower_right.x >= out_size_x){
        cuda::GpuMat roi_right(dst, Rect(cam.upper_left, Point(out_size_x, cam.lower_right.y + 1)));
        cuda::GpuMat roi_right_map_x(cam.map_x, Rect(0, 0, out_size_x - cam.upper_left.x, cam.map_x.size().height));
        cuda::GpuMat roi_right_map_y(cam.map_y, Rect(0, 0, out_size_x - cam.upper_left.x, cam.map_y.size().height));
        cuda::remap(cam.cur_img, roi_right, roi_right_map_x, roi_right_map_y, INTER_LINEAR, BORDER_CONSTANT);
        cuda::GpuMat roi_left(dst, Rect(Point(0, cam.upper_left.y), Point(cam.lower_right.x - out_size_x + 1, cam.lower_right.y + 1)));
        cuda::GpuMat roi_left_map_x(cam.map_x, Rect(Point(out_size_x - cam.upper_left.x, 0), Point(cam.map_x.size())));
        cuda::GpuMat roi_left_map_y(cam.map_y, Rect(Point(out_size_x - cam.upper_left.x, 0), Point(cam.map_y.size())));
        cuda::remap(cam.cur_img, roi_left, roi_left_map_x, roi_left_map_y, INTER_LINEAR, BORDER_CONSTANT);
    }else{
        cuda::GpuMat roi(dst, Rect(cam.upper_left, cam.lower_right + Point(1, 1)));
        cuda::remap(cam.cur_img, roi, cam.map_x, cam.map_y, INTER_LINEAR, BORDER_CONSTANT);
    }
}

void process(cuda::GpuMat &dst, cuda::GpuMat &dst_buffer, cuda::GpuMat &vignette_mask){
    for(int batchi = 0; batchi < cam_batches.size(); batchi++){
        cam_batch &batch = cam_batches[batchi];
        bool write_to_buffer = enable_transparency && batchi > 0;
        for(int cam : batch.cams){
            process_cam(cams[cam], write_to_buffer?dst_buffer:dst, vignette_mask);
        }
        if(write_to_buffer){
            dst_buffer.copyTo(dst, batch.mask);
            dst_buffer.setTo(Scalar(0, 0, 0));
        }
    }
}

void calc_coverage(){
    cuda::GpuMat total_mask(out_size_y, out_size_x, CV_8UC1, Scalar(0));
    for(int i = 0; i < cam_batches.size(); i++){
        cam_batch &batch = cam_batches[i];
        cuda::bitwise_or(total_mask, batch.mask, total_mask);
        if(i == 0 || !enable_transparency){
            batch.mask.release();
        }
    }
    Mat total_mask_cpu(out_size_y, out_size_x, CV_8UC1);
    total_mask.download(total_mask_cpu);
    double area = 0, area_tot = 0;
    double dy = M_PI/out_size_y;
    for(int y = 0; y < out_size_y; y++){
        Vec2d az_alt = inverse_project_equirect(0, y);
        double dx = 2*M_PI*cos(az_alt[1])/out_size_x;
        int i = 0;
        for(int x = 0; x < out_size_x; x++){
            if(total_mask_cpu.at<unsigned char>(y, x) > 0){
                i++;
            }
        }
        area_tot += out_size_x*dx*dy;
        area += i*dx*dy;
        cout << "coverage (at " << az_alt[1]/M_PI*180 << "°): "
             << area/area_tot*100 << "%, precision: " << area_tot/(4*M_PI)*100 << "%" << endl;
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
            if(cam.decoded_img.empty()){
                cam.decoded_img = imread(cam.test_img, IMREAD_COLOR);
                if(cam.decoded_img.empty()){
                    cerr << "failed to read test image " << cam.test_img << endl;
                    exit(1);
                }
            }
        }else{
            cam.cap.retrieve(cam.decoded_img);
        }
        cam.cur_img.upload(cam.decoded_img);
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

void setup_cuda(){
    int cuda_devices_number = cuda::getCudaEnabledDeviceCount();
    for(int i = 0; i < cuda_devices_number; i++){
        cuda::printCudaDeviceInfo(i);
    }
    if(selected_device < cuda_devices_number){
        cuda::setDevice(selected_device);
    }else{
        cout << "selected device index out of range" << endl;
    }
}

void draw_background(Mat dst){
    vector<string> t{"N", "E", "S", "W"};
    double alt = -0.15;
    Scalar c(200, 200, 200);
    for(int i = 0; i < 4; i++){
        double az = M_PI_2*i;
        Point p0(to_pixel(project_equirect(Vec2d(az, alt)), Vec2i(out_size_x, out_size_y)));
        line(dst, p0, p0 + Point(0, 35), c, 5, LINE_AA);
        putText(dst, t[i], p0 + Point(-32, 120), FONT_HERSHEY_COMPLEX, 3, c, 5, LINE_AA);
    }
    for(int i = 0; i < 36; i++){
        double az = 2*M_PI*i/36;
        Point p0(to_pixel(project_equirect(Vec2d(az, alt)), Vec2i(out_size_x, out_size_y)));
        line(dst, p0, p0 + Point(0, 20), c, 3, LINE_AA);
    }
}

int main(int argc, char *argv[]){
    string config = main_config_file;
    if(argc == 2){
        config = argv[1];
    }
    read_config(config);
    setup_cuda();
    cuda::GpuMat vignette_mask;
    if(enable_vignette_correction){
        vignette_mask.create(in_size_y, in_size_x, CV_8UC3);
        precalc_brightness_mask(vignette_mask);
    }
    precalc_pixel_grids();
    if(calculate_coverage){
        calc_coverage();
    }
    if(!use_test_images){
        open_cameras();
    }
    cuda::GpuMat dst(out_size_y, out_size_x, CV_8UC3, Scalar(0, 0, 0));
    cuda::GpuMat dst_buffer;
    if(enable_transparency){
        dst_buffer.create(out_size_y, out_size_x, CV_8UC3);
        dst_buffer.setTo(Scalar(0, 0, 0));
    }
    cuda::GpuMat bg(out_size_y, out_size_x, CV_8UC3, Scalar(0, 0, 0));
    Mat dst_cpu(out_size_y, out_size_x, CV_8UC3, Scalar(0, 0, 0));
    draw_background(dst_cpu);
    bg.upload(dst_cpu);
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
        bg.copyTo(dst);
        process(dst, dst_buffer, vignette_mask);
        dst.download(dst_cpu);
        int64 process_frames_end = getTickCount();
        if(test_mode){
            imwrite(test_output_file, dst_cpu);
            imshow("output", dst_cpu);
            while(waitKey() != 'q');
            destroyAllWindows();
        }
        int64 write_frames_start = 0, write_frames_end = 0;
        if(start_stream){
            if(verbose_mainloop){
                cout << "writing frame" << endl;
            }
            write_frames_start = getTickCount();
            video << dst_cpu;
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
