
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include "face_detector.hpp"
#include "helpers.hpp"

#include <sys/stat.h>
#include <sys/time.h>

using namespace std;
using namespace cv;

enum DATASET { AFW, PASCAL, FDDB, WIDER_TEST, WIDER_VAL, UFDD};
enum MODE {WEBCAM, BENCHMARK_EVALUATION, IMAGE, IMAGE_LIST};

typedef struct _tagMTCNNResult {
	cv::Rect r;
	float score;
}MTCNNResult;

double what_time_is_it_now()
{
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

typedef struct _tagMTCNNCppResult {
	cv::Rect r;
	float score;
}MTCNNCppResult;

std::vector<MTCNNCppResult> mtcnnCppDetection(mtcnn::FaceDetector& fd, cv::Mat img, float min_face_size)
{
	std::vector<MTCNNCppResult> m_results;
	std::vector<mtcnn::Face> faces = fd.detect(img, min_face_size, 0.709f);
	for (size_t i = 0; i < faces.size(); ++i) {
		std::vector<cv::Point> pts;
		for (int p = 0; p < mtcnn::NUM_PTS; ++p) {
			pts.push_back(cv::Point(faces[i].ptsCoords[2 * p], faces[i].ptsCoords[2 * p + 1]));
		}		

		MTCNNCppResult m_result;
		m_result.score = faces[i].score;
		m_result.r = faces[i].bbox.getRect();

		m_results.push_back(m_result);
	}

	for(size_t i=0;i<m_results.size();i++) {
		if(m_results[i].score > 0.7)
			cv::rectangle(img, m_results[i].r, CV_RGB(255,0,0), 2);
	}

	return m_results;
}

void run_webcam(int webcam_id)
{
	mtcnn::FaceDetector fd("../model/", 0.6f, 0.7f, 0.7f, true, false, 0);

	cv::VideoCapture cap(webcam_id);
	if(!cap.isOpened()) {
		cout << "fail to open webcam!" << endl;
		return;
	}

	cv::Mat image;
	while(true) {
	
		cap >> image;

		if(image.empty()) break;
		
		double time_begin = what_time_is_it_now();
		//detect face by min_size(30)
		std::vector<MTCNNCppResult> m_results = mtcnnCppDetection(fd, image, 30.f);
		double time_now = what_time_is_it_now();
		double time_diff = time_now-time_begin;
		cv::imshow("image", image);

		cout << "MTCNNCpp FPS: " << 1/time_diff << endl;

		if (cv::waitKey(1) >= 0) break;		
	}
}

void run_afw_pascal_fddb(DATASET m_data, std::string dataset_path)
{
	std::string str_img_file;
	if(m_data == AFW) str_img_file = "../detections/AFW/afw_img_list.txt";
	else if (m_data == PASCAL) str_img_file = "../detections/PASCAL/pascal_img_list.txt";
	else if (m_data == FDDB) str_img_file = "../detections/FDDB/fddb_img_list.txt";	
	else return ;

	cout << str_img_file << endl;

	std::ifstream inFile(str_img_file.c_str(), std::ifstream::in);

	std::vector<string> image_list;
	std::string imname;
	while(std::getline(inFile, imname)) 
	{
		image_list.push_back(imname);
	}

	std::string str_out_file;
	if(m_data == AFW) str_out_file = "../detections/AFW/mtcnn_afw_dets.txt";
	else if (m_data == PASCAL) str_out_file = "../detections/PASCAL/mtcnn_pascal_dets.txt";
	else if (m_data == FDDB) str_out_file = "../detections/FDDB/mtcnn_fddb_dets.txt";	

	std::ofstream outFile(str_out_file.c_str());

	//initial models without image's width or height
	mtcnn::FaceDetector fd("../model/", 0.05f, 0.05f, 0.05f, true, false, 0);

	// process each image one by one
	for(int i = 0; i < image_list.size(); i++)
	{
		std::string imname = image_list[i];
		std::string tempname = imname;

		if(m_data==AFW) imname = dataset_path+imname+".jpg";
		else if(m_data==PASCAL) imname = dataset_path+imname;
		else if(m_data==FDDB) imname = dataset_path+imname+".jpg";

		cout << "processing image " << i+1 << "/" << image_list.size() << " [" << imname.c_str() << "]" << endl;

		cv::Mat image = cv::imread(imname);

		std::vector<MTCNNCppResult> mtcnn_results = mtcnnCppDetection(fd, image, 25.f);

		if(m_data!=FDDB) 
		{			
			for(int j=0;j<mtcnn_results.size();j++) {
				outFile << tempname << " " << mtcnn_results[j].score << " " << mtcnn_results[j].r.x << " "
					<< mtcnn_results[j].r.y << " " << mtcnn_results[j].r.x+mtcnn_results[j].r.width << " " 
					<< mtcnn_results[j].r.y+mtcnn_results[j].r.height << endl;		
			}			
		} else {
			
			outFile << tempname << "\n";
			outFile << mtcnn_results.size() << "\n";
			for(int j =0; j < mtcnn_results.size(); j++) {
				outFile << mtcnn_results[j].r.x << " " << mtcnn_results[j].r.y << " " 
				<< mtcnn_results[j].r.width << " " << mtcnn_results[j].r.height << " " << mtcnn_results[j].score << "\n";
			}
		}

		imshow("test", image);

		waitKey(1);
	}
	outFile.close();
}

void make_wider_dir(DATASET m_data, std::string& result_dir)
{
	std::vector<std::string> dirs;
	std::ifstream indir("../detections/WIDER/wider_dirs.txt");
	while(!indir.eof()) {
		std::string str_dir;
		indir >> str_dir;		
		if(str_dir=="") continue;
		dirs.push_back(str_dir);
	}

	for(int i =0; i < dirs.size(); i++) 
	{
		int dir_err = mkdir(result_dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
		std::string dir = result_dir+dirs[i];
		dir_err = mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	}
}

void run_wider(DATASET m_data, std::string dataset_path)
{
	std::string result_dir;
	if(m_data == WIDER_TEST) result_dir = "../detections/WIDER/mtcnn_test/";
	else result_dir = "../detections/WIDER/mtcnn_val/";

	make_wider_dir(m_data, result_dir);

	std::string str_img_file;
	if(m_data == WIDER_TEST) str_img_file = "../detections/WIDER/wider_test_list.txt";
	else if (m_data == WIDER_VAL) str_img_file = "../detections/WIDER/wider_val_list.txt";
	else return;

	std::ifstream inFile(str_img_file.c_str(), std::ifstream::in);

	std::vector<string> image_list;
	while(!inFile.eof()) 
	{
		std::string str;
		inFile >> str;
		if(str=="") continue;
		image_list.push_back(str);
	}

	std::string wider_path = dataset_path;

	//initial models without image's width or height
	mtcnn::FaceDetector fd("../model/", 0.1f, 0.1f, 0.1f, true, false, 0);

	for(int i = 0; i < image_list.size(); i++)
	{
		std::string imname = wider_path+image_list[i];
		cout << imname << endl;
		cout << "processing image " << i+1 << "/" << image_list.size() << " [" << image_list[i].c_str() << "]" << endl;

		cv::Mat image = cv::imread(imname);

		std::vector<MTCNNCppResult> mtcnn_results = mtcnnCppDetection(fd, image, 25.f);

		std::string txt_name = image_list[i];
		txt_name.replace(txt_name.end()-4, txt_name.end(), ".txt");
		txt_name = result_dir+txt_name;
	
		std::ofstream out_txt(txt_name.c_str());
		out_txt << image_list[i] << "\n";
		out_txt << mtcnn_results.size() << "\n";
		for(int j =0; j < mtcnn_results.size(); j++) {
			out_txt << mtcnn_results[j].r.x << " " << mtcnn_results[j].r.y << " " << mtcnn_results[j].r.width << " " 
			<< mtcnn_results[j].r.height << " " << mtcnn_results[j].score << "\n";
		}
		out_txt.close();

		imshow("image", image);

		waitKey(1);
	}	
}

void run_ufdd(std::string dataset_path)
{
	std::string result_dir;
	result_dir = "../detections/UFDD/";

	std::string str_img_file;
	str_img_file = "../detections/UFDD/ufdd_img_list.txt";

	std::ifstream inFile(str_img_file.c_str(), std::ifstream::in);

	std::vector<string> image_list;
	while(!inFile.eof()) 
	{
		std::string str;
		inFile >> str;
		if(str=="") continue;
		image_list.push_back(str);
	}

	std::string ufdd_path = dataset_path;

	//initial models without image's width or height
	mtcnn::FaceDetector fd("../model/", 0.1f, 0.1f, 0.1f, true, false, 0);

	for(int i = 0; i < image_list.size(); i++)
	{
		std::string imname = ufdd_path+image_list[i]+".jpg";
		cout << imname << endl;
		cout << "processing image " << i+1 << "/" << image_list.size() << " [" << image_list[i].c_str() << "]" << endl;

		cv::Mat image = cv::imread(imname);

		std::vector<MTCNNCppResult> mtcnn_results = mtcnnCppDetection(fd, image, 25.f);

		std::string txt_name = image_list[i]+".txt";
		txt_name = result_dir+txt_name;
	
		std::ofstream out_txt(txt_name.c_str());
		out_txt << image_list[i] << "\n";
		out_txt << mtcnn_results.size() << "\n";
		for(int j =0; j < mtcnn_results.size(); j++) {
			out_txt << mtcnn_results[j].r.x << " " << mtcnn_results[j].r.y << " " << mtcnn_results[j].r.width << " " 
			<< mtcnn_results[j].r.height << " " << mtcnn_results[j].score << "\n";
		}
		out_txt.close();

		imshow("image", image);

		waitKey(1);
	}	
}

void run_image(std::string image_path)
{
	mtcnn::FaceDetector fd("../model/", 0.6f, 0.7f, 0.7f, true, false, 0);
	
	cv::Mat image = cv::imread(image_path.c_str());
	if(image.empty()) {
		cout << "Image not exist in the specified dir!";
		return;	
	} 

	std::vector<MTCNNCppResult> m_results = mtcnnCppDetection(fd, image, 30.f);

	cv::imshow("image", image);
	waitKey(1);
	getchar();
}

void run_images(std::string image_path)
{
	std::vector<cv::String> img_list;
	glob(image_path+"*.jpg", img_list, false);

	size_t count = img_list.size();
	for(size_t i=0;i<count;i++) 
	{
		mtcnn::FaceDetector fd("../model/", 0.6f, 0.7f, 0.7f, true, false, 0);
		cv::Mat image = cv::imread(img_list[i].c_str());
		if(image.empty()) {
			cout << "Image not exist in the specified dir!";
			return;	
		} 
	
		std::vector<MTCNNCppResult> m_results = mtcnnCppDetection(fd, image, 30.f);

		cv::imshow("image", image);
		cv::waitKey(1);
	}
} 


cv::String keys =
	"{ help h 		| | Print help message. }"
	"{ mode m 		| 0 | Select running mode: "
						"0: WEBCAM - get image streams from webcam, "
						"1: IMAGE - detect faces in single image, "
						"2: IMAGE_LIST - detect faces in set of images, "
						"3: BENCHMARK_EVALUATION - benchmark evaluation, results will be stored in 'detections' }"
	"{ webcam i 	| 0 | webcam id, if mode is 0 }"
	"{ dataset d 	| AFW | select dataset, if mode is 3:"
						   "AFW: afw dataset, "
						   "PASCAL: pascal dataset, "
						   "FDDB: fddb dataset, "
						   "WIDER_VAL: wider validation set, "
						   "WIDER_TEST: wider test set, "
						   "UFDD: ufdd validataion set }"
	"{ path p 		| | Path to image file or image list dir or benchmark dataset }";


int main(int argc, char ** argv) 
{
	cv::CommandLineParser parser(argc, argv, keys);

	if(argc == 1 || parser.has("help")) 
	{
		parser.printMessage();
		return 0;
	}

	MODE m_mode;
	int mode = parser.get<int>("mode");		

	if(mode == 1) m_mode = IMAGE;
	else if(mode == 2) m_mode = IMAGE_LIST;
	else if(mode == 3) m_mode = BENCHMARK_EVALUATION;
	else m_mode = WEBCAM;

	if(m_mode == WEBCAM)
	{
		int webcam_id = parser.get<int>("webcam");
		run_webcam(webcam_id);
	}
	else if(m_mode == IMAGE) 
	{
		std::string image_path = parser.get<String>("path");
		run_image(image_path);
	}
	else if(m_mode == IMAGE_LIST) 
	{
		std::string image_path = parser.get<String>("path");
		run_images(image_path);	
	}
	else if (m_mode == BENCHMARK_EVALUATION)
	{
		DATASET m_data;
		std::string dataset_name = parser.get<String>("dataset");
		cout << dataset_name << endl;
		if(dataset_name == "PASCAL") m_data = PASCAL;
		else if(dataset_name == "FDDB") m_data = FDDB;
		else if(dataset_name == "WIDER_VAL") m_data = WIDER_VAL;
		else if(dataset_name == "WIDER_TEST") m_data = WIDER_TEST;
		else if(dataset_name == "UFDD") m_data = UFDD;
		else if(dataset_name == "AFW") m_data = AFW; 

		std::string dataset_path = parser.get<String>("path");

		if(m_data == AFW || m_data == PASCAL || m_data == FDDB)
			run_afw_pascal_fddb(m_data, dataset_path);
		else if(m_data == WIDER_VAL || m_data == WIDER_TEST) 
			run_wider(m_data, dataset_path);
		else if(m_data == UFDD)
			run_ufdd(dataset_path);
	}

	return 0;
}