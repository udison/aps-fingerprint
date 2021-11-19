package aps.fingerprint.main;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static aps.fingerprint.main.imageproc.FingerprintProcessing.*;
import static aps.fingerprint.main.imageproc.ImgUtils.*;

public class FingerprintRecognition {



  public static void main(String[] args) {
    // Initializes OpenCV
    System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    Imgcodecs codecs = new Imgcodecs();

    // Loads a file
    String file = "D:/DEV/java/fingerprint-aps/assets/fingerprints/fingerprint-1.jpg";
    Mat input = codecs.imread(file, Imgcodecs.IMREAD_GRAYSCALE);
    System.out.println("Image loaded!");
//    showImage("Loaded Image", input);

    // Applies a threshold
    Mat binarized = new Mat();
    Imgproc.threshold(input, binarized, 100, 255, Imgproc.THRESH_BINARY);
    System.out.println("Image binarized!");
//    showImage("Binarized Image", binarized);

    // Thins the image
    Mat thinned = skeleton(binarized.clone());
    System.out.println("Image thinned!");
    saveTest("thinned.jpg", thinned);
//    showImage("Thinned Image", thinned);

    // Cleans some isolated pixels (hit or miss)
    Mat cleaned = hitOrMiss(thinned.clone());
//    showImage("Cleaned Image", cleaned);



    Mat harrisCorner = harrisCorners(cleaned, 125f);
    showImage(harrisCorner);
  }


}
