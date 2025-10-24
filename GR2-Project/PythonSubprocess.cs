using System.Diagnostics;

namespace GR2_Project
{
    internal class PythonSubprocess
    {
        private static Process StartProcess(string scriptPath, string[] args, bool start = true)
        {
            ProcessStartInfo startInfo = new();
            startInfo.FileName = "python";
            startInfo.Arguments = scriptPath;
            foreach (var arg in args)
                startInfo.Arguments += " " + arg;
            startInfo.UseShellExecute = false;
            startInfo.RedirectStandardInput = true;
            startInfo.RedirectStandardOutput = true;
            startInfo.RedirectStandardError = true;
            startInfo.CreateNoWindow = true;
            startInfo.WorkingDirectory = Application.StartupPath + "AI_model\\";

            Process process = new();
            process.StartInfo = startInfo;
            if (start)
                process.Start();
            return process;
        }

        // Thêm scriptName để khởi camera.py hoặc object_box.py
        public static void YoloSubProcess(string scriptName, string mode, PictureBox displayPictureBox, Label fps, Label resolution, string webcamIP, CancellationTokenSource cts, CommandBuffer commandBuffer)
        {
            int frameCount = 0;
            TimeOnly startTime = new();

            if (webcamIP == "")
                webcamIP = "0";

            Process process = StartProcess(scriptName, new string[] { webcamIP, mode });
            Stream stdout = process.StandardOutput.BaseStream;

            StreamReader reader = new(stdout);   // Đọc metadata
            BinaryReader binaryReader = new(stdout); // Đọc ảnh JPEG

            try
            {
                while (!cts.IsCancellationRequested)
                {
                    // Kiểm tra lệnh mới cho task (nếu có)
                    string? command = commandBuffer?.Command;
                    if (!string.IsNullOrEmpty(command))
                    {
                        try
                        {
                            process.StandardInput.WriteLine(command);
                        }
                        catch { }
                    }

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
                        Debug.WriteLine(json);
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
            }
            catch { }
            finally
            {
                try { if (!process.HasExited) process.Kill(); } catch { }
                try { process.Dispose(); } catch { }
            }
        }
    }
}