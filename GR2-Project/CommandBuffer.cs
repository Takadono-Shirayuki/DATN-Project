namespace GR2_Project
{
    internal class CommandBuffer
    {
        public const string BBOX = "bbox";
        public const string POSE = "pose";
        public const string SEG = "seg";

        private string command = string.Empty;
        private Dictionary<string, bool> options = new()
            {
                { BBOX, false },
                { POSE, false },
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
                if (options.ContainsKey(value))
                {
                    options[value] = !options[value];
                    command = value + "_" + (options[value] ? "on" : "off");
                }
                else
                {
                    command = string.Empty;
                }
            }
        }
    }
}
