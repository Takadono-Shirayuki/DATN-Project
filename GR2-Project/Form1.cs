using System.Diagnostics;

namespace GR2_Project
{
    public partial class Form1 : Form
    {
        string webcamIP = "[2402:800:61c7:9fb8:68ea:b2ff:fe8c:41f8]";
        public Form1()
        {
            InitializeComponent();
            Thread yoloThread = new Thread(new ThreadStart(yoloSubProcess));
            yoloThread.IsBackground = true;
            yoloThread.Start();
        }

        private void yoloSubProcess()
        {
            List<byte> tempBuffer = new List<byte>();

            ProcessStartInfo startInfo = new ProcessStartInfo();
            startInfo.FileName = "python";
            startInfo.Arguments = Application.StartupPath + "AI_model/yolo_detect.py " + webcamIP;
            startInfo.UseShellExecute = false;
            startInfo.RedirectStandardOutput = true;
            startInfo.RedirectStandardError = true;
            startInfo.CreateNoWindow = true;

            Process process = new Process();
            process.StartInfo = startInfo;
            process.Start();

            Stream stdout = process.StandardOutput.BaseStream;

            while (true)
            {
                byte[] chunk = new byte[1024];
                int bytesRead = stdout.Read(chunk, 0, chunk.Length);
                if (bytesRead == 0) 
                    continue;

                for (int i = 0; i < bytesRead; i++)
                    tempBuffer.Add(chunk[i]);

                // Tìm điểm kết thúc JPEG
                int endIndex = FindJpegEnd(tempBuffer);
                if (endIndex != -1)
                {
                    int startIndex = FindJpegStart(tempBuffer);
                    if (startIndex != -1 && startIndex < endIndex)
                    {
                        int length = endIndex - startIndex + 1;
                        byte[] data = tempBuffer.GetRange(startIndex, length).ToArray();

                        // Giữ lại phần dư cho ảnh sau
                        tempBuffer.RemoveRange(0, endIndex + 1);

                        // Hiển thị ảnh lên PictureBox
                        using (MemoryStream ms = new MemoryStream(data))
                        {
                            try
                            {
                                Image img = Image.FromStream(ms);
                                pictureBox1.Invoke((MethodInvoker)delegate
                                {
                                    pictureBox1.Image = img;
                                });
                            }
                            catch (Exception ex)
                            {
                                // Xử lý lỗi nếu ảnh không hợp lệ
                                Console.WriteLine("Lỗi khi tạo ảnh từ stream: " + ex.Message);
                            }
                        }
                    }
                }
            }
        }

        // Hàm tìm điểm bắt đầu JPEG (0xFFD8)
        private int FindJpegStart(List<byte> data)
        {
            for (int i = 0; i < data.Count - 1; i++)
                if (data[i] == 0xFF && data[i + 1] == 0xD8)
                    return i;
            return -1;
        }

        // Hàm tìm điểm kết thúc JPEG (0xFFD9)
        private int FindJpegEnd(List<byte> data)
        {
            for (int i = 0; i < data.Count - 1; i++)
                if (data[i] == 0xFF && data[i + 1] == 0xD9)
                    return i + 1;
            return -1;
        }
    }
}