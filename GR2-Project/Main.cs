namespace GR2_Project
{
    public partial class Main : Form
    {
        public Main()
        {
            InitializeComponent();
            EnableControls(false);
        }

        private Task? pythonSubprocess;
        private CancellationTokenSource cts;
        private CommandBuffer commandBuffer;
        public class YoloSubProcessVariables : Classes.YoloSubProcessVariables
        {
            private Label fps;
            private Label resolution;
            private PictureBox displayPictureBox;
            private Label objectID;
            public YoloSubProcessVariables(Label fps, Label resolution, PictureBox displayPictureBox, Label objectID)
            {
                this.fps = fps;
                this.resolution = resolution;
                this.displayPictureBox = displayPictureBox;
                this.objectID = objectID;
            }

            public override double Fps
            {
                get
                {
                    if (fps.Text.Length < 5)
                        return 0;
                    return double.Parse(fps.Text.Substring(5));
                }
                set
                {
                    fps.BeginInvoke(() => fps.Text = "FPS: " + value.ToString("F2"));
                }
            }

            public override int FrameWidth
            {
                get
                {
                    if (resolution.Text.Length < 9)
                        return 0;
                    return int.Parse(resolution.Text.Substring(5).Split('x')[0]);
                }
                set
                {
                    int height = FrameHeight;
                    resolution.BeginInvoke(() => resolution.Text = "Res: " + value + "x" + height);
                }
            }

            public override int FrameHeight
            {
                get
                {
                    if (resolution.Text.Length < 9)
                        return 0;
                    return int.Parse(resolution.Text.Substring(5).Split('x')[1]);
                }
                set
                {
                    int width = FrameWidth;
                    resolution.BeginInvoke(() => resolution.Text = "Res: " + width + "x" + value);
                }
            }

            public override Image Image
            {
                get { return displayPictureBox.Image; }
                set
                {
                    displayPictureBox.BeginInvoke(() =>
                    {
                        displayPictureBox.Image?.Dispose();
                        displayPictureBox.Image = (Image)value.Clone();
                    });
                }
            }

            public override string ObjectID
            {
                get { return objectID.Text; }
                set
                {
                    objectID.BeginInvoke(() => objectID.Text = value);
                }
            }
        }

        private void EnableControls(bool enable)
        {
            button1.Text = "Stop";
            cameraBBox.Enabled = enable;
            cameraPose.Enabled = enable;
            cameraSeg.Enabled = enable;
            
            objectBoxPose.Enabled = enable;
            objectBoxSeg.Enabled = enable;
            if (!enable)
            {
                cameraBBox.Checked = false;
                cameraPose.Checked = false;
                cameraSeg.Checked = false;
                objectBoxPose.Checked = false;
                objectBoxSeg.Checked = false;
                button1.Text = "Start";
            }
        }

        // Bổ sung scriptName để khởi camera.py hoặc object_box.py
        private void StartPythonSubprocess(PictureBox activatedPictureBox, string scriptName)
        {
            Label fps, resolution;
            PictureBox displayPictureBox;
            if (activatedPictureBox == cameraPic)
            {
                fps = cameraFps;
                resolution = CameraResolution;
                displayPictureBox = cameraPic;
            }
            else
            {
                fps = objectBoxFps;
                resolution = objectBoxResolution;
                displayPictureBox = objectBoxPic;
            }
            cts = new();
            commandBuffer = new(scriptName);
            pythonSubprocess = Task.Run(() => PythonSubprocess.YoloSubProcess(scriptName, new YoloSubProcessVariables(cameraFps, CameraResolution, activatedPictureBox, objectID), textBox1.Text, cts, commandBuffer)).ContinueWith(t =>
            {
                if (t.Exception != null)
                {
                    MessageBox.Show("Error in Python subprocess: " + t.Exception.InnerException.Message);
                }
                BeginInvoke(() => EnableControls(false));
            });
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
                if (tabControl1.SelectedTab == tabPage1)
                    StartPythonSubprocess(cameraPic, CommandBuffer.CAMERA_SCRIPT_NAME);
                else if (tabControl1.SelectedTab == tabPage2)
                    StartPythonSubprocess(objectBoxPic, CommandBuffer.CAMERA_SCRIPT_NAME);
            }
            else
            {
                EnableControls(false);
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
            if (pythonSubprocess != null)
            {
                commandBuffer.Command = "cancel";
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
                if (tabControl1.SelectedTab == tabPage1)
                {
                    StartPythonSubprocess(cameraPic, CommandBuffer.CAMERA_SCRIPT_NAME);
                }
                else if (tabControl1.SelectedTab == tabPage2)
                {
                    StartPythonSubprocess(objectBoxPic, CommandBuffer.OBJECT_BOX_SCRIPT_NAME);
                }
            }
        }
    }
}