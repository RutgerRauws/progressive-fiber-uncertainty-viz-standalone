#include <iostream>
#include <vtkPolyDataReader.h>
#include <vtkSmartPointer.h>
#include <vtkGenericDataObjectReader.h>
#include <vtkPolyDataCollection.h>

#include <vtkActor.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>

int main()
{
    std::string inputFileName = "./data/FiberBundle_1_Output Volume-label.vtk";
    
    // Get all data from the file
    vtkSmartPointer<vtkGenericDataObjectReader> reader =
            vtkSmartPointer<vtkGenericDataObjectReader>::New();
    reader->SetFileName(inputFileName.c_str());
    reader->Update();
    
    // All of the standard data types can be checked and obtained like this:
    if(reader->IsFilePolyData())
    {
        std::cout << "output is a polydata" << std::endl;
        vtkPolyData* output = reader->GetPolyDataOutput();
        std::cout << "output has " << output->GetNumberOfLines() << " lines." << std::endl;
    }
    
    return EXIT_SUCCESS;
}