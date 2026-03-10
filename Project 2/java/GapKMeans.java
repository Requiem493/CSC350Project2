import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;
import java.util.Scanner;

/**
 * AugmentedVersionKMeans.java - a JUnit-testable interface for the Model AI
 * Assignments k-Means Clustering exercises.
 * Includes Exercise 2 (iterated k-means) and Exercise 3 (Gap Statistic).
 *
 * @author Todd W. Neller
 */
public class GapKMeans {
    private int dim; // the number of dimensions in the data
    private int k, kMin, kMax; // the allowable range of the of clusters
    private int iter; // the number of k-Means Clustering iterations per k
    private double[][] data; // the data vectors for clustering
    private double[][] centroids; // the cluster centroids
    private int[] clusters; // assigned clusters for each data point
    private Random random = new Random();

    /**
     * Read the specified data input format from the given file and return a
     * double[][] with each row being a data point and each column being a dimension
     * of the data.
     *
     * @param filename the data input source file
     * @return a double[][] with each row being a data point and each column being a
     *         dimension of the data
     */
    public double[][] readData(String filename) {
        int numPoints = 0;

        try {
            Scanner in = new Scanner(new File(filename));
            try {
                dim = Integer.parseInt(in.nextLine().split(" ")[1]);
                numPoints = Integer.parseInt(in.nextLine().split(" ")[1]);
            } catch (Exception e) {
                System.err.println("Invalid data file format. Exiting.");
                e.printStackTrace();
                System.exit(1);
            }
            double[][] data = new double[numPoints][dim];
            for (int i = 0; i < numPoints; i++) {
                String line = in.nextLine();
                Scanner lineIn = new Scanner(line);
                for (int j = 0; j < dim; j++)
                    data[i][j] = lineIn.nextDouble();
                lineIn.close();
            }
            in.close();
            return data;
        } catch (FileNotFoundException e) {
            System.err.println("Could not locate source file. Exiting.");
            e.printStackTrace();
            System.exit(1);
        }
        return null;
    }

    /**
     * Set the given data as the clustering data as a double[][] with each row being
     * a data point and each column being a dimension of the data.
     *
     * @param data the given clustering data
     */
    public void setData(double[][] data) {
        this.data = data;
        this.dim = data[0].length;
    }

    /**
     * Return the clustering data as a double[][] with each row being a data point
     * and each column being a dimension of the data.
     *
     * @return the clustering data
     */
    public double[][] getData() {
        return data;
    }

    /**
     * Return the number of dimensions of the clustering data.
     *
     * @return the number of dimensions of the clustering data
     */
    public int getDim() {
        return dim;
    }

    /**
     * Set the minimum and maximum allowable number of clusters k. If a single given
     * k is to be used, then kMin == kMax. If kMin &lt; kMax, then all k from kMin
     * to kMax inclusive will be compared using the gap statistic. The minimum WCSS
     * run of the k with the maximum gap will be the result.
     *
     * @param kMin minimum number of clusters
     * @param kMax maximum number of clusters
     */
    public void setKRange(int kMin, int kMax) {
        this.kMin = kMin;
        this.kMax = kMax;
        this.k = kMin;
    }

    /**
     * Return the number of clusters k. After calling kMeansCluster() with a range
     * from kMin to kMax, this value will be the k yielding the maximum gap
     * statistic.
     *
     * @return the number of clusters k.
     */
    public int getK() {
        return k;
    }

    /**
     * Set the number of iterations to perform k-Means Clustering and choose the
     * minimum WCSS result.
     *
     * @param iter the number of iterations to perform k-Means Clustering
     */
    public void setIter(int iter) {
        this.iter = iter;
    }

    /**
     * Return the array of centroids indexed by cluster number and centroid
     * dimension.
     *
     * @return the array of centroids indexed by cluster number and centroid
     *         dimension.
     */
    public double[][] getCentroids() {
        return centroids;
    }

    /**
     * Return a parallel array of cluster assignments such that data[i] belongs to
     * the cluster clusters[i] with centroid centroids[clusters[i]].
     *
     * @return a parallel array of cluster assignments
     */
    public int[] getClusters() {
        return clusters;
    }

    /**
     * Return the Euclidean distance between the two given point vectors.
     *
     * @param p1 point vector 1
     * @param p2 point vector 2
     * @return the Euclidean distance between the two given point vectors
     */
    private double getDistance(double[] p1, double[] p2) {
        double sumOfSquareDiffs = 0;
        for (int i = 0; i < p1.length; i++) {
            double diff = p1[i] - p2[i];
            sumOfSquareDiffs += diff * diff;
        }
        return Math.sqrt(sumOfSquareDiffs);
    }

    /**
     * Return the Within-Clusters Sum-of-Squares measure for the current clustering.
     * Uses the global data, clusters, and centroids fields.
     *
     * @return the Within-Clusters Sum-of-Squares measure
     */
    public double getWCSS() {
        double wcss = 0;
        for (int i = 0; i < data.length; i++) {
            double dist = getDistance(data[i], centroids[clusters[i]]);
            wcss += dist * dist;
        }
        return wcss;
    }

    /**
     * Compute the WCSS for an arbitrary dataset using the current k, clusters,
     * and centroids. Used internally during Gap Statistic reference dataset runs.
     *
     * @param dataset the dataset to compute WCSS for
     * @return the WCSS for the given dataset
     */
    private double getWCSSForDataset(double[][] dataset) {
        double wcss = 0;
        for (int i = 0; i < dataset.length; i++) {
            double dist = getDistance(dataset[i], centroids[clusters[i]]);
            wcss += dist * dist;
        }
        return wcss;
    }

    /**
     * Assign each data point to the nearest centroid and return whether or not any
     * cluster assignments changed.
     *
     * @return whether or not any cluster assignments changed
     */
    public boolean assignNewClusters() {
        boolean changed = false;
        for (int i = 0; i < data.length; i++) {
            int nearest = 0;
            double minDist = getDistance(data[i], centroids[0]);
            for (int j = 1; j < k; j++) {
                double dist = getDistance(data[i], centroids[j]);
                if (dist < minDist) {
                    minDist = dist;
                    nearest = j;
                }
            }
            if (clusters[i] != nearest) {
                clusters[i] = nearest;
                changed = true;
            }
        }
        return changed;
    }

    /**
     * Assign each point in the given dataset to the nearest centroid.
     * Updates the global clusters array. Used for reference dataset runs.
     *
     * @param dataset the dataset whose points are being assigned
     * @return whether any assignment changed
     */
    private boolean assignNewClustersForDataset(double[][] dataset) {
        boolean changed = false;
        for (int i = 0; i < dataset.length; i++) {
            int nearest = 0;
            double minDist = getDistance(dataset[i], centroids[0]);
            for (int j = 1; j < k; j++) {
                double dist = getDistance(dataset[i], centroids[j]);
                if (dist < minDist) {
                    minDist = dist;
                    nearest = j;
                }
            }
            if (clusters[i] != nearest) {
                clusters[i] = nearest;
                changed = true;
            }
        }
        return changed;
    }

    /**
     * Compute new centroids at the mean point of each cluster of points.
     * Uses the global data field.
     */
    public void computeNewCentroids() {
        centroids = new double[k][dim]; // reset to zeros
        int[] counts = new int[k];

        for (int i = 0; i < data.length; i++) {
            int c = clusters[i];
            counts[c]++;
            for (int j = 0; j < dim; j++)
                centroids[c][j] += data[i][j];
        }

        for (int i = 0; i < k; i++)
            for (int j = 0; j < dim; j++)
                centroids[i][j] /= counts[i];
    }

    /**
     * Compute new centroids at the mean point of each cluster for a given dataset.
     * Used for reference dataset runs during Gap Statistic calculation.
     *
     * @param dataset the dataset to compute centroids for
     */
    private void computeNewCentroidsForDataset(double[][] dataset) {
        centroids = new double[k][dim];
        int[] counts = new int[k];

        for (int i = 0; i < dataset.length; i++) {
            int c = clusters[i];
            counts[c]++;
            for (int j = 0; j < dim; j++)
                centroids[c][j] += dataset[i][j];
        }

        for (int i = 0; i < k; i++)
            for (int j = 0; j < dim; j++)
                centroids[i][j] /= counts[i];
    }

    /**
     * Run one full k-means clustering pass on the given dataset using Forgy
     * initialization, updating the global clusters and centroids fields.
     * Returns the WCSS of the resulting clustering.
     *
     * @param dataset the dataset to cluster
     * @return the WCSS of the clustering result
     */
    private double runKMeansOnDataset(double[][] dataset) {
        // Forgy initialization: pick k random distinct points as initial centroids
        clusters = new int[dataset.length];
        ArrayList<Integer> indices = new ArrayList<Integer>();
        for (int j = 0; j < dataset.length; j++)
            indices.add(j);
        Collections.shuffle(indices, random);
        centroids = new double[k][dim];
        for (int j = 0; j < k; j++)
            centroids[j] = dataset[indices.get(j)].clone();

        // Iterate: assign clusters, recompute centroids, until stable
        boolean changed;
        do {
            changed = assignNewClustersForDataset(dataset);
            computeNewCentroidsForDataset(dataset);
        } while (changed);

        return getWCSSForDataset(dataset);
    }

    /**
     * Perform k-means clustering with Forgy initialization.
     *
     * - If kMin == kMax: runs iter iterations and picks the result with lowest WCSS
     * (Exercise 2 behaviour).
     *
     * - If kMin &lt; kMax: uses the Gap Statistic to automatically choose the best
     * k
     * (Exercise 3 behaviour). For each k in [kMin, kMax]:
     * 1. Run iter iterations on real data; take log of the minimum WCSS.
     * 2. Generate 100 random reference datasets with the same size, sampling
     * each dimension uniformly within that dimension's [min, max] range.
     * Run 1 k-means pass on each; average their log(WCSS) values.
     * 3. Gap(k) = avgRandLogWCSS - logMinWCSS.
     * The k with the largest Gap is selected as the final k.
     */
    public void kMeansCluster() {
        // Track the best result across all k values
        double bestGap = Double.NEGATIVE_INFINITY;
        double bestWCSS = Double.MAX_VALUE;
        int bestK = kMin;
        int[] bestClusters = null;
        double[][] bestCentroids = null;

        // --- Step 1: Compute per-dimension min/max for reference dataset generation
        // ---
        double[] dimMin = new double[dim];
        double[] dimMax = new double[dim];
        for (int d = 0; d < dim; d++) {
            dimMin[d] = Double.MAX_VALUE;
            dimMax[d] = Double.MIN_VALUE;
        }
        for (int i = 0; i < data.length; i++) {
            for (int d = 0; d < dim; d++) {
                if (data[i][d] < dimMin[d])
                    dimMin[d] = data[i][d];
                if (data[i][d] > dimMax[d])
                    dimMax[d] = data[i][d];
            }
        }

        // --- Step 2: Loop over every k in [kMin, kMax] ---
        for (int kCurrent = kMin; kCurrent <= kMax; kCurrent++) {
            this.k = kCurrent;

            // --- Step 2a: Run iter iterations on real data (Exercise 2 logic) ---
            double iterBestWCSS = Double.MAX_VALUE;
            int[] iterBestClusters = null;
            double[][] iterBestCentroids = null;

            for (int i = 0; i < iter; i++) {
                // Forgy initialization on real data
                clusters = new int[data.length];
                ArrayList<Integer> indices = new ArrayList<Integer>();
                for (int j = 0; j < data.length; j++)
                    indices.add(j);
                Collections.shuffle(indices, random);
                centroids = new double[k][dim];
                for (int j = 0; j < k; j++)
                    centroids[j] = data[indices.get(j)].clone();

                boolean changed;
                do {
                    changed = assignNewClusters();
                    computeNewCentroids();
                } while (changed);

                double wcss = getWCSS();
                if (wcss < iterBestWCSS) {
                    iterBestWCSS = wcss;
                    iterBestClusters = clusters.clone();
                    iterBestCentroids = centroids.clone();
                }
            }

            // log of the minimum WCSS found across all iter runs for this k
            double logMinWCSS = Math.log(iterBestWCSS);

            // --- Step 2b: Gap Statistic — only computed when kMin < kMax ---
            if (kMin < kMax) {
                // Generate 100 random reference datasets and average their log(WCSS)
                int numRefDatasets = 100;
                double sumRandLogWCSS = 0.0;

                for (int r = 0; r < numRefDatasets; r++) {
                    // Build a random dataset: same number of points, uniform in each dimension
                    double[][] refData = new double[data.length][dim];
                    for (int i = 0; i < data.length; i++)
                        for (int d = 0; d < dim; d++)
                            refData[i][d] = dimMin[d] + random.nextDouble() * (dimMax[d] - dimMin[d]);

                    // Run ONE k-means pass on the reference dataset
                    double refWCSS = runKMeansOnDataset(refData);
                    sumRandLogWCSS += Math.log(refWCSS);
                }

                double avgRandLogWCSS = sumRandLogWCSS / numRefDatasets;

                // Gap(k) = average log WCSS on random data minus log WCSS on real data
                // A large gap means real data clusters much more tightly than random
                double gap = avgRandLogWCSS - logMinWCSS;

                // Keep this k if it has the highest gap so far
                if (gap > bestGap) {
                    bestGap = gap;
                    bestK = kCurrent;
                    bestClusters = iterBestClusters;
                    bestCentroids = iterBestCentroids;
                }
            } else {
                // kMin == kMax: no gap statistic needed, just keep the best WCSS result
                if (iterBestWCSS < bestWCSS) {
                    bestWCSS = iterBestWCSS;
                    bestK = kCurrent;
                    bestClusters = iterBestClusters;
                    bestCentroids = iterBestCentroids;
                }
            }
        }

        // --- Step 3: Restore global state to the best k found ---
        this.k = bestK;
        this.clusters = bestClusters;
        this.centroids = bestCentroids;

        // Restore data reference so getWCSS() works correctly after the method returns
        // (runKMeansOnDataset temporarily uses reference datasets; reset data pointer)
    }

    /**
     * Export cluster data in the given data output format to the file provided.
     *
     * @param filename the destination file
     */
    public void writeClusterData(String filename) {
        try {
            FileWriter out = new FileWriter(filename);

            out.write(String.format("%% %d dimensions\n", dim));
            out.write(String.format("%% %d points\n", data.length));
            out.write(String.format("%% %d clusters/centroids\n", k));
            out.write(String.format("%% %f within-cluster sum of squares\n", getWCSS()));
            for (int i = 0; i < k; i++) {
                out.write(i + " ");
                for (int j = 0; j < dim; j++)
                    out.write(centroids[i][j] + (j < dim - 1 ? " " : "\n"));
            }
            for (int i = 0; i < data.length; i++) {
                out.write(clusters[i] + " ");
                for (int j = 0; j < dim; j++)
                    out.write(data[i][j] + (j < dim - 1 ? " " : "\n"));
            }
            out.flush();
            out.close();
        } catch (Exception e) {
            System.err.println("Error writing to file");
            e.printStackTrace();
            System.exit(1);
        }
    }

    /**
     * Read UNIX-style command line parameters to specify the type of k-Means
     * Clustering algorithm applied to the formatted data specified.
     * "-k int" specifies both the minimum and maximum number of clusters.
     * "-kmin int" specifies the minimum number of clusters.
     * "-kmax int" specifies the maximum number of clusters.
     * "-iter int" specifies the number of times k-Means Clustering is performed.
     * "-in filename" specifies the source file for input data.
     * "-out filename" specifies the destination file for cluster data.
     *
     * @param args command-line parameters
     */
    public static void main(String[] args) {
        int kMin = 2, kMax = 2, iter = 1;
        ArrayList<String> attributes = new ArrayList<String>();
        ArrayList<Integer> values = new ArrayList<Integer>();
        int i = 0;
        String infile = null;
        String outfile = null;
        while (i < args.length) {
            if (args[i].equals("-k") || args[i].equals("-kmin") || args[i].equals("-kmax") || args[i].equals("-iter")) {
                attributes.add(args[i++].substring(1));
                if (i == args.length) {
                    System.err.println("No integer value for " + attributes.get(attributes.size() - 1) + ".");
                    System.exit(1);
                }
                try {
                    values.add(Integer.parseInt(args[i]));
                    i++;
                } catch (Exception e) {
                    System.err.printf("Error parsing \"%s\" as an integer.", args[i]);
                    System.exit(2);
                }
            } else if (args[i].equals("-in")) {
                i++;
                if (i == args.length) {
                    System.err.println("No string value provided for input source");
                    System.exit(1);
                }
                infile = args[i];
                i++;
            } else if (args[i].equals("-out")) {
                i++;
                if (i == args.length) {
                    System.err.println("No string value provided for output source");
                    System.exit(1);
                }
                outfile = args[i];
                i++;
            } else {
                i++; // skip unrecognised args
            }
        }

        for (i = 0; i < attributes.size(); i++) {
            String attribute = attributes.get(i);
            if (attribute.equals("k"))
                kMin = kMax = values.get(i);
            else if (attribute.equals("kmin"))
                kMin = values.get(i);
            else if (attribute.equals("kmax"))
                kMax = values.get(i);
            else if (attribute.equals("iter"))
                iter = values.get(i);
        }

        AugmentedVersionKMeans km = new AugmentedVersionKMeans();
        km.setKRange(kMin, kMax);
        km.setIter(iter);
        km.setData(km.readData(infile));
        km.kMeansCluster();
        km.writeClusterData(outfile);
    }
}
