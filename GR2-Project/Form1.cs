namespace GR2_Project
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
            EnableControls(false);
        }

        private Task? pythonSubprocess;
        private CancellationTokenSource cts = new();
        private CommandBuffer commandBuffer = new();
        private void EnableControls(bool enable)
        {
            bbox.Enabled = enable;
            pose.Enabled = enable;
            seg.Enabled = enable;
            if (!enable)
            {
                bbox.Checked = false;
                pose.Checked = false;
                seg.Checked = false;
            }
        }

        private void StartPythonSubprocess()
        {
            cts = new();

            pythonSubprocess = new(() => PythonSubprocess.YoloSubProcess(pictureBox1, fps, resolution, textBox1.Text, cts, commandBuffer));
            pythonSubprocess.Start();
        }

        private void StopPythonSubprocess()
        {
            cts.Cancel();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            if (button1.Text == "Start")
            {
                EnableControls(true);
                button1.Text = "Stop";
                StartPythonSubprocess();
            }
            else
            {
                EnableControls(false);
                button1.Text = "Start";
                StopPythonSubprocess();
            }
        }

        private void checkBox1_CheckedChanged(object sender, EventArgs e)
        {
            commandBuffer.Command = CommandBuffer.BBOX;
        }

        private void checkBox2_CheckedChanged(object sender, EventArgs e)
        {
            commandBuffer.Command = CommandBuffer.POSE;
        }

        private void checkBox3_CheckedChanged(object sender, EventArgs e)
        {
            commandBuffer.Command = CommandBuffer.SEG;
        }
    }
}