#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <CL\cl.hpp>
#include "Utils.h"

void print_help() {
	std::cerr << "Application usage:" << std::endl;
	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int avgTemp(cl::Program program, cl::Buffer buffer_A, cl::Buffer buffer_B, cl::CommandQueue queue,
	size_t vector_size, size_t vector_elements, vector<int> outputFloat, size_t local_size)
{
	// Setup and execute the kernel (i.e. device code)
	cl::Kernel kernel_Average = cl::Kernel(program, "avgTemp");//Average

	kernel_Average.setArg(0, buffer_A);
	kernel_Average.setArg(1, buffer_B);
	kernel_Average.setArg(2, cl::Local(1));

	queue.enqueueNDRangeKernel(kernel_Average, cl::NullRange, cl::NDRange(vector_elements), cl::NDRange(local_size));

	//Copy the result from device to host
	queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, vector_size, &outputFloat[0]);

	return outputFloat[0];
}


int maxTemperature(cl::Program program, cl::Buffer buffer_A, cl::Buffer buffer_B, cl::CommandQueue queue,
	size_t vector_size, size_t vector_elements, vector<int> outputFloat, size_t local_size)
{
	// Setup and execute the kernel (i.e. device code)
	cl::Kernel kernel_Max = cl::Kernel(program, "maxTemperature");//Maximum
	kernel_Max.setArg(0, buffer_A);
	kernel_Max.setArg(1, buffer_B);
	kernel_Max.setArg(2, cl::Local(1));

	queue.enqueueNDRangeKernel(kernel_Max, cl::NullRange, cl::NDRange(vector_elements), cl::NDRange(local_size));

	//Copy the result from device to host
	queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, vector_size, &outputFloat[0]);

	return outputFloat[0];
}

int minTemperature(cl::Program program, cl::Buffer buffer_A, cl::Buffer buffer_B, cl::CommandQueue queue,
	size_t vector_size, size_t vector_elements, vector<int> outputFloat, size_t local_size)
{
	// Setup and execute the kernel (i.e. device code)
	cl::Kernel kernel_Min = cl::Kernel(program, "minTemperature");//Minimum

	kernel_Min.setArg(0, buffer_A);
	kernel_Min.setArg(1, buffer_B);
	kernel_Min.setArg(2, cl::Local(1));

	queue.enqueueNDRangeKernel(kernel_Min, cl::NullRange, cl::NDRange(vector_elements), cl::NDRange(local_size));

	//Copy the result from device to host
	queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, vector_size, &outputFloat[0]);

	return outputFloat[0];
}

vector<int> hist_simple(cl::Program program, cl::Buffer buffer_A, cl::Buffer buffer_B, cl::CommandQueue queue,
	size_t vector_size, size_t vector_elements, vector<int> outputFloat, size_t local_size, int count, int minValueal, int maxValueal)
{
	cl::Kernel kernel_Hist = cl::Kernel(program, "hist_simple");

	kernel_Hist.setArg(0, buffer_A);
	kernel_Hist.setArg(1, buffer_B);
	kernel_Hist.setArg(2, count);
	kernel_Hist.setArg(3, minValueal);
	kernel_Hist.setArg(4, maxValueal);


	queue.enqueueNDRangeKernel(kernel_Hist, cl::NullRange, cl::NDRange(vector_elements), cl::NDRange(local_size));

	queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, vector_size, &outputFloat[0]);

	return outputFloat;
}

int main(int argc, char **argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;

	for (int i = 1; i < argc; i++)	{
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); }
	}

	//detect any potential exceptions
	try {
		//Part 2 - host operations
		//2.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context);

		//2.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "my_kernels.cl");

		cl::Program program(context, sources);

		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		std::vector<int> A(10, 1);

		//All variables used to manage data entered from text file and store in relevant vectors
		string line;
		char delim = ' ';
		string inputData = "";
		int numOfSpace = 0;
		vector<int> temperature;
		vector<int> month;
		int temp = 0; //Used to store a temporary value of temp as atomic_add only works on integers.
		int temp1 = 0;
		
		//read in data
		ifstream myfile("temp_lincolnshire_short.txt");
		if (myfile.is_open())
		{
			cout << "Reading in data...\n";
			while (getline(myfile, line))
			{
				for (int i = 0; i < line.length(); i++)
				{
					inputData += line[i];
					if (line[i] == delim || i == line.length() - 1)
					{
							
						numOfSpace++;
						switch (numOfSpace)
						{
						//get every 6th value, as this is temperature
						case 6: 
							temp = int(stof(inputData) * 10);
								temperature.push_back(temp);
								numOfSpace = 0;
								//reset to 0
								inputData = "";
								break;
																
						default: 
							inputData = "";
							break;
						}
					}
				}
			}
			myfile.close();
		}
		
		size_t vector_elements = temperature.size();//number of elements
		size_t vector_size = temperature.size()*sizeof(int);//size in bytes
		size_t lsize = 32;
		size_t local_size = ( 64, 1 );
		size_t padding_size = temperature.size() % local_size;

		//if the input vector is not a multiple of the local_size
		//insert additional neutral elements (0 for addition) so that the total will not be affected (make work for my working set of data)
		if (padding_size) {
			//create an extra vector with neutral values
			std::vector<int> temperature_ext(local_size - padding_size, 1000);
			//append that extra vector to our input
			temperature.insert(temperature.end(), temperature_ext.begin(), temperature_ext.end());
		}

		//host - output
		std::vector<int> outputList(vector_elements);
		std::vector<int> histOutput(vector_elements);
		size_t output_size = histOutput.size()*sizeof(int);//size in bytes

		//device - buffers
		cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, vector_size);
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, vector_size);
		cl::Buffer buffer_output_size(context, CL_MEM_READ_WRITE, output_size);

		//Part 5 - device operations

		//5.1 Copy arrays A and B to device memory
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, vector_size, &temperature[0]);
		queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, vector_size, &outputList[0]);
		queue.enqueueFillBuffer(buffer_output_size, 0, 0, output_size);//zero B buffer on device memory

		float maxValue = 0; //float for max, as needs converting
		float minValue = 0; //float for min, as needs converting
		float output = 0.0f;

		output = (float)(avgTemp(program, buffer_A, buffer_B, queue, vector_size, vector_elements, outputList, local_size));
		//divide by 10 to give true decimal value
		output /= 10.0f;
		output = output / vector_elements;
		std::cout << "Average temperature: " << output << std::endl;
		output = (float)(minTemperature(program, buffer_A, buffer_B, queue, vector_size, vector_elements, outputList, local_size));
		minValue = output;
		output /= 10.0f;
		std::cout << "Minimum temperature: " << output << std::endl;
		output = (float)(maxTemperature(program, buffer_A, buffer_B, queue, vector_size, vector_elements, outputList, local_size));
		maxValue = output;
		output /= 10.0f;
		std::cout << "Maximum temperature: " << output << std::endl;
		//request number of bins from user
		std::cout << "Number of Bins required?"<<endl;
		int binNum;
		cin >> binNum;
		//validate that input is not less than 0
		while (cin.fail() || binNum <= 0)
		{
			std::cout << "Enter a number greater than 0 \n";
			cin >> binNum;
		}
		outputList = (hist_simple(program, buffer_A, buffer_output_size, queue, vector_size, vector_elements, outputList, local_size, binNum, minValue, maxValue));
        float increment = ((maxValue - minValue) / binNum);
        cout << "increment size is " << (increment/10) << std::endl;
		//display bins
		for (int i = 1; i < binNum+1; i++) {
			std::cout << "(" << i - 1 << ")" << (outputList[i-1]) << std::endl;
		}		
	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}

	std::cin.get(); //get response

	std::cin.get(); //ensure program does not close before user wants it to
	return 0;
}