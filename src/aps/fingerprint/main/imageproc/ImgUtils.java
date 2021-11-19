package aps.fingerprint.main.imageproc;

import org.opencv.core.Mat;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;

public class ImgUtils {
  public static void saveTest(String fileName, Mat output) {
    String outputFile = "D:/DEV/java/fingerprint-aps/assets/tests/" + fileName;
    Imgcodecs.imwrite(outputFile, output);
  }

  public static void showImage(Mat img) {
    HighGui.imshow("Image", img);
    HighGui.waitKey();
  }

  public static void showImage(String title, Mat img) {
    HighGui.imshow(title, img);
    HighGui.waitKey();
  }
}
