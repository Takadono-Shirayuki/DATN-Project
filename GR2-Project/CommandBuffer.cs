using System.Diagnostics;

namespace GR2_Project
{
    internal class CommandBuffer
    {
        public const string BBOX = "bbox";
        public const string POSE = "pose";
        public const string SEG = "seg";
        public const string CAMERA_MODE = "camera";
        public const string OBJECT_BOX_MODE = "object_box";

        private string mode;
        public CommandBuffer(string mode)
        {
            this.mode = mode;
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
                if (mode == CAMERA_MODE)
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
                else if (mode == OBJECT_BOX_MODE)
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
