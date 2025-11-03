using GR2_Project.Classes;
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
        public static void YoloSubProcess(string scriptName, YoloSubProcessVariables yoloSubProcessVariables, string webcamIP, CancellationTokenSource cts, CommandBuffer commandBuffer)
        {
            int frameCount = 0;
            TimeOnly startTime = new();

            if (webcamIP == "")
                webcamIP = "0";

            Process process = StartProcess(scriptName, new string[] { webcamIP });
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
                        bool end = meta.RootElement.TryGetProperty("end", out _);
                        if (end)
                            break; // Kết thúc nếu có trường "end"
                        string objectID = meta.RootElement.TryGetProperty("object_id", out var objIdProp) ? objIdProp.GetString() ?? "" : "";
                        yoloSubProcessVariables.ObjectID = objectID;
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
                            yoloSubProcessVariables.Image = img;
                            if (frameCount == 1)
                            {
                                startTime = TimeOnly.FromDateTime(DateTime.Now);
                                yoloSubProcessVariables.FrameWidth = img.Width;
                                yoloSubProcessVariables.FrameHeight = img.Height;
                            }
                            else if (frameCount % 10 == 0)
                            {
                                TimeSpan elapsed = TimeOnly.FromDateTime(DateTime.Now).ToTimeSpan() - startTime.ToTimeSpan();
                                yoloSubProcessVariables.Fps = frameCount / elapsed.TotalSeconds;
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