namespace GR2_Project
{
    partial class Main
    {
        /// <summary>
        ///  Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        ///  Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        ///  Required method for Designer support - do not modify
        ///  the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            cameraPic = new PictureBox();
            textBox1 = new TextBox();
            panel1 = new Panel();
            button1 = new Button();
            label1 = new Label();
            panel2 = new Panel();
            groupBox1 = new GroupBox();
            resolution = new Label();
            fps = new Label();
            seg = new CheckBox();
            pose = new CheckBox();
            bbox = new CheckBox();
            tabControl1 = new TabControl();
            tabPage1 = new TabPage();
            tabPage2 = new TabPage();
            objectBoxPic = new PictureBox();
            panel3 = new Panel();
            objectID = new Label();
            button2 = new Button();
            button3 = new Button();
            ((System.ComponentModel.ISupportInitialize)cameraPic).BeginInit();
            panel1.SuspendLayout();
            panel2.SuspendLayout();
            groupBox1.SuspendLayout();
            tabControl1.SuspendLayout();
            tabPage1.SuspendLayout();
            tabPage2.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)objectBoxPic).BeginInit();
            panel3.SuspendLayout();
            SuspendLayout();
            // 
            // cameraPic
            // 
            cameraPic.Dock = DockStyle.Fill;
            cameraPic.Location = new Point(253, 3);
            cameraPic.Margin = new Padding(5, 4, 5, 4);
            cameraPic.Name = "cameraPic";
            cameraPic.Size = new Size(1136, 550);
            cameraPic.SizeMode = PictureBoxSizeMode.Zoom;
            cameraPic.TabIndex = 0;
            cameraPic.TabStop = false;
            // 
            // textBox1
            // 
            textBox1.Location = new Point(159, 8);
            textBox1.Margin = new Padding(5, 4, 5, 4);
            textBox1.Name = "textBox1";
            textBox1.Size = new Size(560, 36);
            textBox1.TabIndex = 1;
            textBox1.Text = "[2402:800:61c7:9fb8:68ea:b2ff:fe8c:41f8]";
            // 
            // panel1
            // 
            panel1.BorderStyle = BorderStyle.Fixed3D;
            panel1.Controls.Add(button1);
            panel1.Controls.Add(label1);
            panel1.Controls.Add(textBox1);
            panel1.Dock = DockStyle.Top;
            panel1.Location = new Point(0, 0);
            panel1.Margin = new Padding(5, 4, 5, 4);
            panel1.Name = "panel1";
            panel1.Size = new Size(1400, 54);
            panel1.TabIndex = 2;
            // 
            // button1
            // 
            button1.Location = new Point(727, 8);
            button1.Name = "button1";
            button1.Size = new Size(149, 36);
            button1.TabIndex = 3;
            button1.Text = "Start";
            button1.UseVisualStyleBackColor = true;
            button1.Click += button1_Click;
            // 
            // label1
            // 
            label1.AutoSize = true;
            label1.Location = new Point(14, 11);
            label1.Margin = new Padding(5, 0, 5, 0);
            label1.Name = "label1";
            label1.Size = new Size(135, 29);
            label1.TabIndex = 2;
            label1.Text = "Webcam IP:";
            // 
            // panel2
            // 
            panel2.BorderStyle = BorderStyle.Fixed3D;
            panel2.Controls.Add(groupBox1);
            panel2.Controls.Add(seg);
            panel2.Controls.Add(pose);
            panel2.Controls.Add(bbox);
            panel2.Dock = DockStyle.Left;
            panel2.Location = new Point(3, 3);
            panel2.Name = "panel2";
            panel2.Size = new Size(250, 550);
            panel2.TabIndex = 3;
            // 
            // groupBox1
            // 
            groupBox1.Controls.Add(resolution);
            groupBox1.Controls.Add(fps);
            groupBox1.Dock = DockStyle.Bottom;
            groupBox1.Location = new Point(0, 421);
            groupBox1.Name = "groupBox1";
            groupBox1.Size = new Size(246, 125);
            groupBox1.TabIndex = 3;
            groupBox1.TabStop = false;
            groupBox1.Text = "Camera Info";
            // 
            // resolution
            // 
            resolution.AutoSize = true;
            resolution.Location = new Point(14, 80);
            resolution.Name = "resolution";
            resolution.Size = new Size(0, 29);
            resolution.TabIndex = 1;
            // 
            // fps
            // 
            fps.AutoSize = true;
            fps.Location = new Point(14, 40);
            fps.Name = "fps";
            fps.Size = new Size(0, 29);
            fps.TabIndex = 0;
            // 
            // seg
            // 
            seg.AutoSize = true;
            seg.Location = new Point(14, 100);
            seg.Name = "seg";
            seg.Size = new Size(185, 33);
            seg.TabIndex = 2;
            seg.Text = "Phân đoạn ảnh";
            seg.UseVisualStyleBackColor = true;
            seg.CheckedChanged += checkBox3_CheckedChanged;
            // 
            // pose
            // 
            pose.AutoSize = true;
            pose.Location = new Point(14, 60);
            pose.Name = "pose";
            pose.Size = new Size(175, 33);
            pose.TabIndex = 1;
            pose.Text = "Khung xương";
            pose.UseVisualStyleBackColor = true;
            pose.CheckedChanged += checkBox2_CheckedChanged;
            // 
            // bbox
            // 
            bbox.AutoSize = true;
            bbox.Location = new Point(14, 20);
            bbox.Name = "bbox";
            bbox.Size = new Size(165, 33);
            bbox.TabIndex = 0;
            bbox.Text = "Hộp giới hạn";
            bbox.UseVisualStyleBackColor = true;
            bbox.CheckedChanged += checkBox1_CheckedChanged;
            // 
            // tabControl1
            // 
            tabControl1.Controls.Add(tabPage1);
            tabControl1.Controls.Add(tabPage2);
            tabControl1.Dock = DockStyle.Fill;
            tabControl1.Location = new Point(0, 54);
            tabControl1.Name = "tabControl1";
            tabControl1.SelectedIndex = 0;
            tabControl1.Size = new Size(1400, 598);
            tabControl1.TabIndex = 4;
            tabControl1.SelectedIndexChanged += tabControl1_SelectedIndexChanged;
            // 
            // tabPage1
            // 
            tabPage1.Controls.Add(cameraPic);
            tabPage1.Controls.Add(panel2);
            tabPage1.Location = new Point(4, 38);
            tabPage1.Name = "tabPage1";
            tabPage1.Padding = new Padding(3);
            tabPage1.Size = new Size(1392, 556);
            tabPage1.TabIndex = 0;
            tabPage1.Text = "Camera";
            tabPage1.UseVisualStyleBackColor = true;
            // 
            // tabPage2
            // 
            tabPage2.Controls.Add(objectBoxPic);
            tabPage2.Controls.Add(panel3);
            tabPage2.Location = new Point(4, 38);
            tabPage2.Name = "tabPage2";
            tabPage2.Padding = new Padding(3);
            tabPage2.Size = new Size(1392, 556);
            tabPage2.TabIndex = 1;
            tabPage2.Text = "Hộp đối tượng";
            tabPage2.UseVisualStyleBackColor = true;
            // 
            // objectBoxPic
            // 
            objectBoxPic.Dock = DockStyle.Fill;
            objectBoxPic.Location = new Point(253, 3);
            objectBoxPic.Name = "objectBoxPic";
            objectBoxPic.Size = new Size(1136, 550);
            objectBoxPic.TabIndex = 4;
            objectBoxPic.TabStop = false;
            // 
            // panel3
            // 
            panel3.Controls.Add(objectID);
            panel3.Controls.Add(button2);
            panel3.Controls.Add(button3);
            panel3.Dock = DockStyle.Left;
            panel3.Location = new Point(3, 3);
            panel3.Name = "panel3";
            panel3.Size = new Size(250, 550);
            panel3.TabIndex = 5;
            // 
            // objectID
            // 
            objectID.AutoSize = true;
            objectID.Location = new Point(9, 13);
            objectID.Name = "objectID";
            objectID.Size = new Size(0, 29);
            objectID.TabIndex = 4;
            // 
            // button2
            // 
            button2.Location = new Point(3, 512);
            button2.Name = "button2";
            button2.Size = new Size(112, 33);
            button2.TabIndex = 3;
            button2.Text = "<<";
            button2.UseVisualStyleBackColor = true;
            // 
            // button3
            // 
            button3.Location = new Point(127, 512);
            button3.Name = "button3";
            button3.Size = new Size(117, 33);
            button3.TabIndex = 2;
            button3.Text = ">>";
            button3.UseVisualStyleBackColor = true;
            // 
            // Main
            // 
            AutoScaleDimensions = new SizeF(14F, 29F);
            AutoScaleMode = AutoScaleMode.Font;
            ClientSize = new Size(1400, 652);
            Controls.Add(tabControl1);
            Controls.Add(panel1);
            Font = new Font("Times New Roman", 15F, FontStyle.Regular, GraphicsUnit.Point, 0);
            Margin = new Padding(5, 4, 5, 4);
            Name = "Main";
            Text = "Form1";
            ((System.ComponentModel.ISupportInitialize)cameraPic).EndInit();
            panel1.ResumeLayout(false);
            panel1.PerformLayout();
            panel2.ResumeLayout(false);
            panel2.PerformLayout();
            groupBox1.ResumeLayout(false);
            groupBox1.PerformLayout();
            tabControl1.ResumeLayout(false);
            tabPage1.ResumeLayout(false);
            tabPage2.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)objectBoxPic).EndInit();
            panel3.ResumeLayout(false);
            panel3.PerformLayout();
            ResumeLayout(false);
        }

        #endregion

        private PictureBox cameraPic;
        private TextBox textBox1;
        private Panel panel1;
        private Button button1;
        private Label label1;
        private Panel panel2;
        private CheckBox bbox;
        private CheckBox seg;
        private CheckBox pose;
        private GroupBox groupBox1;
        private Label resolution;
        private Label fps;
        private TabControl tabControl1;
        private TabPage tabPage1;
        private TabPage tabPage2;
        private PictureBox objectBoxPic;
        private Button button3;
        private Panel panel3;
        private Label objectID;
        private Button button2;
    }
}
