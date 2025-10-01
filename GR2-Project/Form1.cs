using System.Diagnostics;

namespace GR2_Project
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        private Task? pythonSubprocess;
        private CancellationTokenSource cts = new();
        private void StartPythonSubprocess()
        {
            cts = new();
            pythonSubprocess = new(() => PythonSubprocess.YoloSubProcess(pictureBox1, textBox1.Text, cts));
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
                button1.Text = "Stop";
                StartPythonSubprocess();
            }
            else
            {
                button1.Text = "Start";
                StopPythonSubprocess();
            }    
        }
    }
}