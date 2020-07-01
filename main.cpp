#include <iostream>
#include <chrono>
#include <thread>

#include <vtkSmartPointer.h>
#include <vtkGenericDataObjectReader.h>
#include <vtkPolyDataCollection.h>

#include <vtkActor.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkPolyDataMapper.h>
#include <vtkLine.h>
#include <vtkInteractorStyleTrackballCamera.h>

const std::string INPUT_FILE_NAME = "./data/FiberBundle_1_Output Volume-label.vtk"; //temporary hardcoded input file

vtkSmartPointer<vtkPolyData> fiberPolyData;
vtkSmartPointer<vtkCellArray> polyLines;
vtkSmartPointer<vtkRenderWindow> renderWindow;

vtkSmartPointer<vtkPolyData> readPolyData(const std::string& inputFileName);

void add_fibers_thread()
{
    fiberPolyData->GetLines()->InitTraversal();
    vtkSmartPointer<vtkIdList> idList = vtkSmartPointer<vtkIdList>::New();

    while(fiberPolyData->GetLines()->GetNextCell(idList))
    {
        //std::cout << "Line has " << idList->GetNumberOfIds() << " points." << std::endl;

        polyLines->InsertNextCell(idList);
        polyLines->Modified();
        renderWindow->Render();

        std::cout << "Rendered line!" << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

int main()
{
    std::cout << "Application started." << std::endl;
    
    try {
        fiberPolyData = readPolyData(INPUT_FILE_NAME);
        std::cout << "Input has " << fiberPolyData->GetNumberOfLines() << " fibers." << std::endl;
    }
    catch( const std::invalid_argument& e ) {
        std::cout << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    vtkSmartPointer<vtkPolyData> progressivePolyData  = vtkSmartPointer<vtkPolyData>::New();
    polyLines = vtkSmartPointer<vtkCellArray>::New();
    progressivePolyData->SetLines(polyLines);
    progressivePolyData->SetPoints(fiberPolyData->GetPoints());
    
//    vtkSmartPointer<vtkConeSource> cone = vtkSmartPointer<vtkConeSource>::New();
    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
//    mapper->SetInputConnection(cone->GetOutputPort());
    mapper->SetInputData(progressivePolyData);
    
    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);
    
    vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
    renderer->AddActor(actor);
    renderer->SetBackground(0, 0, 0);
    
    renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->SetSize(800, 600);
    renderWindow->AddRenderer(renderer);
    
    vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor =
            vtkSmartPointer<vtkRenderWindowInteractor>::New();
    renderWindowInteractor->SetRenderWindow(renderWindow);
    
    vtkSmartPointer<vtkInteractorStyleTrackballCamera> style =
            vtkSmartPointer<vtkInteractorStyleTrackballCamera>::New();
    renderWindowInteractor->SetInteractorStyle(style);
    
    renderer->ResetCamera();
    renderWindow->Render();
    renderWindowInteractor->Initialize();
    
    std::thread addFibers_t(add_fibers_thread);
    renderWindowInteractor->Start();
    addFibers_t.join();
    
    return EXIT_SUCCESS;
}

vtkSmartPointer<vtkPolyData> readPolyData(const std::string& inputFileName)
{
    vtkSmartPointer<vtkGenericDataObjectReader> reader = vtkSmartPointer<vtkGenericDataObjectReader>::New();
    reader->SetFileName(inputFileName.c_str());
    reader->Update();
    
    if(!reader->IsFilePolyData())
    {
        throw std::invalid_argument("The file input is not polygon data");
    }
    
    vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
    polyData->CopyStructure(reader->GetPolyDataOutput());
    
    return polyData;
}