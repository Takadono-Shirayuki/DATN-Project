using System.Threading.Tasks;

namespace GR2_Project
{
    public partial class Main : Form
    {
        private const string CAMERA_MODE = CommandBuffer.CAMERA_MODE;
        private const string OBJECT_BOX_MODE = CommandBuffer.OBJECT_BOX_MODE;

        public Main()
        {
            InitializeComponent();
            EnableControls(false);
        }

        private Task? pythonSubprocess;
        private CancellationTokenSource cts;
        private CommandBuffer commandBuffer;
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

        // Bổ sung scriptName để khởi camera.py hoặc object_box.py
        private void StartPythonSubprocess(PictureBox activatedPictureBox, string mode, string scriptName)
        {
            cts = new();
            commandBuffer = new(mode);
            pythonSubprocess = Task.Run(() => PythonSubprocess.YoloSubProcess(scriptName, mode, activatedPictureBox, fps, resolution, textBox1.Text, cts, commandBuffer));
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
                if (tabControl1.SelectedTab == tabPage1)
                    StartPythonSubprocess(cameraPic, CAMERA_MODE, "camera.py");
                else if (tabControl1.SelectedTab == tabPage2)
                    StartPythonSubprocess(objectBoxPic, OBJECT_BOX_MODE, "object_box.py");
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

        // Khi đổi tab: gửi "cancel", chờ tối đa 2s cho tiến trình kết thúc, nếu không thì cancel token và khởi mới
        private async void tabControl1_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (tabControl1.SelectedTab == tabPage1)
            {
                if (pythonSubprocess != null)
                {
                    // gửi lệnh yêu cầu Python thoát
                    try { commandBuffer.Command = "cancel"; } catch { }
                    // chờ tối đa 2s
                    var finished = await Task.WhenAny(pythonSubprocess, Task.Delay(2000));
                    if (finished != pythonSubprocess)
                    {
                        // ép hủy nếu chưa xong
                        try { cts.Cancel(); } catch { }
                        try { await pythonSubprocess; } catch { }
                    }
                    else
                    {
                        try { await pythonSubprocess; } catch { }
                    }
                    pythonSubprocess = null;
                }
                StartPythonSubprocess(cameraPic, CAMERA_MODE, "camera.py");
            }
            else if (tabControl1.SelectedTab == tabPage2)
            {
                if (pythonSubprocess != null)
                {
                    try { commandBuffer.Command = "cancel"; } catch { }
                    var finished = await Task.WhenAny(pythonSubprocess, Task.Delay(2000));
                    if (finished != pythonSubprocess)
                    {
                        try { cts.Cancel(); } catch { }
                        try { await pythonSubprocess; } catch { }
                    }
                    else
                    {
                        try { await pythonSubprocess; } catch { }
                    }
                    pythonSubprocess = null;
                }
                StartPythonSubprocess(objectBoxPic, OBJECT_BOX_MODE, "object_box.py");
            }
        }
    }
}