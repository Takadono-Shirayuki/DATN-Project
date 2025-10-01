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
            startInfo.RedirectStandardOutput = true;
            startInfo.RedirectStandardError = true;
            startInfo.CreateNoWindow = true;
            Process process = new();
            process.StartInfo = startInfo;
            if (start)
                process.Start();
            return process;
        }

        public static void YoloSubProcess(PictureBox displayPictureBox, string webcamIP, CancellationTokenSource cts)
        {
            Process process = StartProcess("yolo_subprocess.py", new string[] { webcamIP });
            Stream stdout = process.StandardOutput.BaseStream;

            List<List<float>> detections = new(); // Bounding boxes
            List<List<List<float>>> keypoints = new(); // Keypoints cho từng người

            StreamReader reader = new(stdout);   // Đọc metadata
            BinaryReader binaryReader = new(stdout); // Đọc ảnh JPEG

            while (!cts.IsCancellationRequested)
            {
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
                detections.Clear();
                keypoints.Clear();

                try
                {
                    var meta = System.Text.Json.JsonDocument.Parse(json);
                    imageSize = meta.RootElement.GetProperty("size").GetInt32();

                    // Detections: danh sách bounding boxes
                    foreach (var box in meta.RootElement.GetProperty("detections").EnumerateArray())
                    {
                        List<float> coords = new();
                        foreach (var coord in box.EnumerateArray())
                            coords.Add(coord.GetSingle());
                        detections.Add(coords);
                    }

                    // Keypoints: danh sách các điểm khớp
                    foreach (var person in meta.RootElement.GetProperty("keypoints").EnumerateArray())
                    {
                        List<List<float>> personKeypoints = new();
                        foreach (var point in person.EnumerateArray())
                        {
                            List<float> kp = new();
                            foreach (var value in point.EnumerateArray())
                                kp.Add(value.GetSingle());
                            personKeypoints.Add(kp); // [x, y, conf]
                        }
                        keypoints.Add(personKeypoints);
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine("Lỗi khi phân tích metadata: " + ex.Message);
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
                        displayPictureBox.Invoke((MethodInvoker)delegate
                        {
                            displayPictureBox.Image = img;
                        });

                        // 👉 Bạn có thể xử lý detections và keypoints ở đây
                        // Ví dụ: vẽ khung người và khung xương lên overlay
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine("Lỗi khi tạo ảnh từ stream: " + ex.Message);
                    }
                }
            }
        }
    }
}
