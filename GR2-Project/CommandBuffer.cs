using System.Diagnostics;

namespace GR2_Project
{
    internal class CommandBuffer
    {
        public const string BBOX = "bbox";
        public const string POSE = "pose";
        public const string SEG = "seg";
        public const string CAMERA_SCRIPT_NAME = "camera.py";
        public const string OBJECT_BOX_SCRIPT_NAME = "object_box.py";

        private string scriptName;
        public CommandBuffer(string scriptName)
        {
            this.scriptName = scriptName;
        }

        private string command = string.Empty;
        private Dictionary<string, bool> cameraOptions = new()
            {
                { BBOX, false },
                { POSE, false },
                { SEG, false }
            };
        private Dictionary<string, bool> objectBoxOptions = new()
            {
                { SEG, false }
            };

        public string Command
        {
            get
            {
                string temp = command;
                command = string.Empty; // Clear after get
                return temp;
            }
            set
            {
                if (scriptName == CAMERA_SCRIPT_NAME)
                {
                    if (cameraOptions.ContainsKey(value))
                    {
                        cameraOptions[value] = !cameraOptions[value];
                        command = value + "_" + (cameraOptions[value] ? "on" : "off");
                    }
                    else
                    {
                        command = string.Empty;
                    }
                }
                else if (scriptName == OBJECT_BOX_SCRIPT_NAME)
                {
                    if (objectBoxOptions.ContainsKey(value))
                    {
                        objectBoxOptions[value] = !objectBoxOptions[value];
                        command = value + "_" + (objectBoxOptions[value] ? "on" : "off");
                    }
                    else
                    {
                        command = string.Empty;
                    }
                }
                else
                {
                    command = string.Empty;
                }
            }
        }
    } 
}
