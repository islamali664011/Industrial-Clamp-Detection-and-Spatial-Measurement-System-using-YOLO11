from roboflow import Roboflow

rf = Roboflow(api_key="Hidden_ROBOFLOW_API_KEY")

project = rf.workspace("your-workspace").project("your-project")

version = project.version(3)

dataset = version.download("yolov11")
