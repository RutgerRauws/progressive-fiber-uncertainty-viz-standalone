#include <iostream>
#include <vtkPolyDataReader.h>
#include <vtkSmartPointer.h>
#include <vtkGenericDataObjectReader.h>
#include <vtkPolyDataCollection.h>

#include <vtkActor.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>

vtkSmartPointer<vtkPolyData> readPolyData(const std::string& inputFileName);

int main()
{
    std::string inputFileName = "./data/FiberBundle_1_Output Volume-label.vtk";
    std::cout << "Application started." << std::endl;
    
    vtkSmartPointer<vtkPolyData> output;
    
    try {
        output = readPolyData(inputFileName);
        std::cout << "output has " << output->GetNumberOfLines() << " lines." << std::endl;
    }
    catch( const std::invalid_argument& e ) {
        std::cout << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}

vtkSmartPointer<vtkPolyData> readPolyData(const std::string& inputFileName)
{
    vtkSmartPointer<vtkGenericDataObjectReader> reader = vtkSmartPointer<vtkGenericDataObjectReader>::New();
    reader->SetFileName(inputFileName.c_str());
    reader->Update();
    
    if(!reader->IsFilePolyData())
    {
        throw new std::invalid_argument("The file input is not polygon data");
    }
    
    vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
    polyData->CopyStructure(reader->GetPolyDataOutput());
    
    return polyData;
}