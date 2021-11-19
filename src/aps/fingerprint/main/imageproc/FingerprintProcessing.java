package aps.fingerprint.main.imageproc;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class FingerprintProcessing {
  public static Mat skeleton(Mat _img) {
    Mat img = _img.clone();

    Mat skel = new Mat(img.size(), CvType.CV_8UC1, new Scalar(0));
    Mat temp = new Mat();
    Mat eroded = new Mat();
    Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_CROSS, new Size(3, 3));

    boolean done;

    do {
      Imgproc.erode(img, eroded, element);
      Imgproc.dilate(eroded, temp, element);
      Core.subtract(img, temp, temp);
      Core.bitwise_or(skel, temp, skel);
      eroded.copyTo(img);

      done = (Core.countNonZero(img) == 0);
    } while (!done);

    return skel;
  }

  public static Mat hitOrMiss(Mat _img) {
    Mat img = _img.clone();
    Mat img_comp = new Mat();
    Mat hitormiss1 = new Mat();
    Mat hitormiss2 = new Mat();
    Mat hitormiss = new Mat();

    Mat kernel1 = new Mat(3, 3, CvType.CV_8UC1) {
      {
        put(0, 0, 0);
        put(0, 1, 0);
        put(0, 2, 0);

        put(1, 0, 0);
        put(1, 1, 1);
        put(1, 2, 0);

        put(2, 0, 0);
        put(2, 1, 0);
        put(2, 2, 0);
      }
    };

    Mat kernel2 = new Mat(3, 3, CvType.CV_8UC1) {
      {
        put(0, 0, 1);
        put(0, 1, 1);
        put(0, 2, 1);

        put(1, 0, 1);
        put(1, 1, 0);
        put(1, 2, 1);

        put(2, 0, 1);
        put(2, 1, 1);
        put(2, 2, 1);
      }
    };

    Core.bitwise_not(img, img_comp);

    Imgproc.morphologyEx(img, hitormiss1, Imgproc.MORPH_ERODE, kernel1);
    Imgproc.morphologyEx(img_comp, hitormiss2, Imgproc.MORPH_ERODE, kernel2);
    Core.bitwise_and(hitormiss1, hitormiss2, hitormiss);

    Mat hitormiss_comp = new Mat();
    Mat del_isolated = new Mat();

    Core.bitwise_not(hitormiss, hitormiss_comp);
    Core.bitwise_and(img, img, del_isolated, hitormiss_comp);

    return del_isolated;
  }

  public static Mat harrisCorners(Mat _img, float threshold) {
    Mat img = _img.clone();
    Mat harrisCorners = new Mat();
    Mat harrisNormalised = new Mat();

    Imgproc.cornerHarris(img, harrisCorners, 2, 3, 0.04, Core.BORDER_DEFAULT);
    Core.normalize(harrisCorners, harrisNormalised, 0, 255, Core.NORM_MINMAX, CvType.CV_32FC1, new Mat());

    List<KeyPoint> keyPoints = new ArrayList<>();
    Mat rescaled = new Mat();

    Core.convertScaleAbs(harrisNormalised, rescaled);

    Mat harrisC = new Mat(rescaled.size(), CvType.CV_8UC3);
    int[] fromTo = { 0,0, 1,1, 2,2 };

    MatOfInt fromToMat = new MatOfInt();
    fromToMat.fromArray(fromTo);

    Core.mixChannels(Arrays.asList(rescaled, rescaled, rescaled), Arrays.asList(harrisC), fromToMat);

    for (int x = 0; x < harrisNormalised.cols(); x++) {
      for (int y = 0; y < harrisNormalised.rows(); y++) {
        if((int)harrisNormalised.get(y, x)[0] > threshold) {
          Imgproc.circle(harrisC, new Point(x, y), 5, new Scalar(0, 255, 0), 1);
          Imgproc.circle(harrisC, new Point(x, y), 1, new Scalar(0, 0, 255), 1);
          keyPoints.add(new KeyPoint(x, y, 1));
        }
      }
    }

    return harrisC;
  }
}
