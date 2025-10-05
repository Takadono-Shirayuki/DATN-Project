using System.Diagnostics;

namespace GR2_Project
{
    internal class PythonSubprocess
    {
        private static Process StartProcess(string scriptPath, string[] args, bool start = true)
        {
            ProcessStartInfo startInfo = new();
            startInfo.FileName = "python";
            startInfo.Arguments = Application.StartupPath + "AI_model\\" + scriptPath;
            foreach (var arg in args)
                startInfo.Arguments += " " + arg;
            startInfo.UseShellExecute = false;
            startInfo.RedirectStandardInput = true;
            startInfo.RedirectStandardOutput = true;
            startInfo.RedirectStandardError = true;
            startInfo.CreateNoWindow = true;
            Process process = new();
            process.StartInfo = startInfo;
            if (start)
                process.Start();
            return process;
        }

        public static void YoloSubProcess(PictureBox displayPictureBox, Label fps, Label resolution, string webcamIP, CancellationTokenSource cts, CommandBuffer commandBuffer)
        {
            int frameCount = 0;
            TimeOnly startTime = new();

            Process process = StartProcess("yolo_detect.py", new string[] { webcamIP });
            Stream stdout = process.StandardOutput.BaseStream;

            StreamReader reader = new(stdout);   // Đọc metadata
            BinaryReader binaryReader = new(stdout); // Đọc ảnh JPEG

            while (!cts.IsCancellationRequested)
            {
                // Kiểm tra lệnh mới cho task (nếu có)
                string? command = commandBuffer.Command;
                if (!string.IsNullOrEmpty(command))
                    process.StandardInput.WriteLine(command);

                // Đọc dòng đầu tiên để kiểm tra metadata
                string line = reader.ReadLine();
                if (line == null || line != "--META--")
                    continue;

                // Đọc metadata JSON
                string json = "";
                while ((line = reader.ReadLine()) != null && line != "--ENDMETA--")
                {
                    json += line;
                }

                // Giải mã metadata
                int imageSize = 0;
                try
                {
                    var meta = System.Text.Json.JsonDocument.Parse(json);
                    imageSize = meta.RootElement.GetProperty("size").GetInt32();
                }
                catch
                {
                    continue;
                }

                // Đọc đúng số byte của ảnh JPEG
                byte[] imageData = binaryReader.ReadBytes(imageSize);
                if (imageData.Length != imageSize)
                    continue;

                // Hiển thị ảnh lên PictureBox
                using (MemoryStream ms = new(imageData))
                {
                    try
                    {
                        Image img = Image.FromStream(ms);
                        frameCount++;
                        displayPictureBox.Invoke((MethodInvoker)delegate
                        {
                            displayPictureBox.Image = img;
                        });
                        if (frameCount == 1)
                        {
                            startTime = TimeOnly.FromDateTime(DateTime.Now);
                            resolution.Invoke((MethodInvoker)delegate
                            {
                                resolution.Text = $"{img.Width}x{img.Height}";
                            });
                        }
                        else if (frameCount % 10 == 0)
                        {
                            TimeSpan elapsed = TimeOnly.FromDateTime(DateTime.Now).ToTimeSpan() - startTime.ToTimeSpan();
                            double fpsValue = frameCount / elapsed.TotalSeconds;
                            fps.Invoke((MethodInvoker)delegate
                            {
                                fps.Text = $"FPS: {fpsValue:F2}";
                            });
                        }
                    }
                    catch { }
                }
            }
            process.Kill();
        }
    }
}