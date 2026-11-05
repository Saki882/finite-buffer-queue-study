// ref_sim_v3.cpp
// Finite-buffer FIFO tail-drop discrete-event simulator.
//
// Arrivals:
//   h=1 : Poisson arrivals
//   h>1 : two-state MMPP with mean arrival rate fixed by rho
//
// Service:
//   exp   : exponential service time with mean packetSize/C
//   const : constant service time with mean packetSize/C
//   corr  : two-point packet sizes (s1,s2), driven by a two-state Markov chain.
//           phi is the lag-1 correlation of the state indicator.
//
// K includes the packet in service.
//
// Important measurement definitions:
//   meanQ     : time-average number in the system over [warm,T], including service.
//   fullFrac  : fraction of [warm,T] for which the system is full.
//   loss run  : consecutive lost arrivals in the observed arrival sequence.
//   recovery  : time from the first loss in an episode until occupancy <= floor(theta*K).
//   delay     : completed sojourn times of packets ACCEPTED in [warm,T] only.
//               Accepted packets still present at T are right-censored and counted separately.
//   Little    : diagnostic ratio using the arrival-cohort delay sample; boundary censoring can
//               make it differ from one. Do not treat it as a proof of stationarity.
//
// Random-number compatibility:
//   The queue, arrival, service and loss logic retain the original main RNG draw order.
//   Delay-reservoir sampling uses a separate RNG. Additional measurement bookkeeping does not
//   consume random numbers, so legacy queue metrics remain comparable for identical parameters
//   and seeds.
//
// Build:
//   g++ -O2 -std=c++17 -Wall -Wextra -pedantic ref_sim_v3.cpp -o ref_sim_v3
//
// Examples:
//   ./ref_sim_v3 --rho=0.9 --K=20 --mode=exp --simTime=2000 --warm=100 --reps=5
//   ./ref_sim_v3 --rho=0.9 --K=20 --mode=corr --h=5 --soj=0.05 --phi=0.6 --reps=10 --csv=run.csv
//
// CSV policy: the program creates a NEW CSV file and refuses to overwrite or append to an
// existing file. Use a different filename for each configuration to keep headers consistent.

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

struct Params {
    std::string mode = "const";
    double rho = 0.9, h = 1.0, soj = 0.05, phi = 0.0;
    double s1 = 100.0, s2 = 1100.0, C = 10e6;
    double T = 200.0, warm = 10.0, theta = 0.5;
    int K = 20, maxHist = 10;
    unsigned long long seed = 1;
};

struct Lag1 {
    long long n = 0;
    double sx = 0.0, sxx = 0.0, sxy = 0.0, prev = 0.0;
    bool has = false;

    void add(double x) {
        ++n;
        sx += x;
        sxx += x * x;
        if (has) sxy += prev * x;
        prev = x;
        has = true;
    }

    double value() const {
        if (n < 3) return std::numeric_limits<double>::quiet_NaN();
        const double mean = sx / static_cast<double>(n);
        const double variance = sxx / static_cast<double>(n) - mean * mean;
        // Autocorrelation is undefined for a constant-valued sequence.
        if (variance <= std::numeric_limits<double>::epsilon() *
                           std::max(1.0, mean * mean))
            return std::numeric_limits<double>::quiet_NaN();
        return (sxy / static_cast<double>(n - 1) - mean * mean) / variance;
    }
};

struct Packet {
    double arrival = 0.0;
    bool inMeasurementCohort = false;
};

struct Result {
    long long off = 0, lost = 0;
    long long runs = 0, runsGE2 = 0, lostInGE2 = 0, maxRun = 0;
    int lossRunCensored = 0;
    std::vector<long long> hist;

    double p = 0.0, meanRun = 0.0, burst = 0.0;
    double meanQ = 0.0, fullFrac = 0.0;

    long long fullN = 0;
    int fullCens = 0;
    double fullMeanDur = std::numeric_limits<double>::quiet_NaN();
    double fullCensoredDuration = std::numeric_limits<double>::quiet_NaN();
    double fullCensDur = std::numeric_limits<double>::quiet_NaN();

    long long recN = 0;
    int recCens = 0;
    double recCensDur = std::numeric_limits<double>::quiet_NaN();
    double recMean = std::numeric_limits<double>::quiet_NaN();
    double recMed = std::numeric_limits<double>::quiet_NaN();
    double recP95 = std::numeric_limits<double>::quiet_NaN();
    double recMax = std::numeric_limits<double>::quiet_NaN();

    long long delayCompleted = 0, delayCensored = 0;
    double delayMean = std::numeric_limits<double>::quiet_NaN();
    double delayP95 = std::numeric_limits<double>::quiet_NaN();
    double little = std::numeric_limits<double>::quiet_NaN();

    double acfInter = std::numeric_limits<double>::quiet_NaN();
    double acfSvc = std::numeric_limits<double>::quiet_NaN();
    int threshold = 0;
};

static double percentile(std::vector<double> values, double q) {
    if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
    std::sort(values.begin(), values.end());
    if (values.size() == 1) return values.front();

    // Linear interpolation between adjacent order statistics.
    const double pos = q * static_cast<double>(values.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(pos));
    const size_t hi = static_cast<size_t>(std::ceil(pos));
    const double frac = pos - static_cast<double>(lo);
    return values[lo] * (1.0 - frac) + values[hi] * frac;
}

static double parseDouble(const std::string& value, const std::string& name) {
    if (value.empty()) throw std::runtime_error("empty value for --" + name);
    errno = 0;
    char* end = nullptr;
    const double x = std::strtod(value.c_str(), &end);
    if (errno == ERANGE || end == value.c_str() || *end != '\0' || !std::isfinite(x))
        throw std::runtime_error("invalid numeric value for --" + name + ": " + value);
    return x;
}

static long long parseInteger(const std::string& value, const std::string& name) {
    if (value.empty()) throw std::runtime_error("empty value for --" + name);
    errno = 0;
    char* end = nullptr;
    const long long x = std::strtoll(value.c_str(), &end, 10);
    if (errno == ERANGE || end == value.c_str() || *end != '\0')
        throw std::runtime_error("invalid integer value for --" + name + ": " + value);
    return x;
}

static unsigned long long parseSeed(const std::string& value) {
    if (value.empty() || value[0] == '-')
        throw std::runtime_error("--seed must be a non-negative integer");
    errno = 0;
    char* end = nullptr;
    const unsigned long long x = std::strtoull(value.c_str(), &end, 10);
    if (errno == ERANGE || end == value.c_str() || *end != '\0')
        throw std::runtime_error("invalid integer value for --seed: " + value);
    return x;
}

static Result simulate(const Params& P, unsigned long long seed) {
    std::mt19937_64 rng(seed);  // Main RNG: retain draw sequence for legacy metrics.
    std::mt19937_64 rngDelay(seed ^ 0x9E3779B97F4A7C15ULL);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    auto Exp = [&](double mean) { return -mean * std::log(1.0 - U(rng)); };

    const double meanSize = 0.5 * (P.s1 + P.s2);
    const double meanSvc = meanSize * 8.0 / P.C;
    const double lambda = P.rho / meanSvc;
    const double lambdaLow = 2.0 * lambda / (1.0 + P.h);
    const double lambdaHigh = P.h * lambdaLow;
    const double stateLambda[2] = {lambdaLow, lambdaHigh};

    // For the symmetric two-state chain, the state-switch probability is (1-phi)/2.
    const double pSwitch = 0.5 * (1.0 - P.phi);
    const int K = P.K;
    const double T = P.T, warm = P.warm;

    int arrivalState = (U(rng) < 0.5) ? 1 : 0;
    int sizeState = (U(rng) < 0.5) ? 1 : 0;
    double nextStateSwitch = Exp(P.soj);

    auto nextArrival = [&](double from) {
        for (;;) {
            const double gap = Exp(1.0 / stateLambda[arrivalState]);
            if (from + gap < nextStateSwitch) return from + gap;
            from = nextStateSwitch;
            arrivalState = 1 - arrivalState;
            nextStateSwitch = from + Exp(P.soj);
        }
    };

    double tArrival = nextArrival(0.0);
    double tDeparture = std::numeric_limits<double>::infinity();

    Result R;
    R.hist.assign(static_cast<size_t>(P.maxHist + 1), 0);
    R.threshold = static_cast<int>(std::floor(P.theta * K));

    int n = 0;
    double lastEvent = 0.0, areaQ = 0.0, areaFull = 0.0;
    long long currentLossRun = 0;

    bool recoveryActive = false;
    double recoveryStart = 0.0;
    std::vector<double> recoveryDurations;

    bool fullActive = false;
    double fullStart = 0.0, fullDurationSum = 0.0;

    std::deque<Packet> fifo;

    // Reservoir sample for delay percentiles; exact mean is accumulated separately.
    const size_t RESERVOIR_CAPACITY = 1000000;
    std::vector<double> delayReservoir;
    delayReservoir.reserve(RESERVOIR_CAPACITY);
    unsigned long long delaySeen = 0;
    double delaySum = 0.0;

    Lag1 interarrivalAcf, serviceAcf;
    double previousArrival = -1.0;

    auto finishLossRun = [&](long long length) {
        if (length <= 0) return;
        const size_t bin = static_cast<size_t>(std::min<long long>(length, P.maxHist));
        ++R.hist[bin];
        R.maxRun = std::max(R.maxRun, length);
        if (length >= 2) {
            ++R.runsGE2;
            R.lostInGE2 += length;
        }
    };

    auto advanceTime = [&](double t) {
        const double from = std::max(lastEvent, warm);
        if (t > from) {
            areaQ += static_cast<double>(n) * (t - from);
            if (n == K) areaFull += t - from;
        }
        lastEvent = t;
    };

    auto startService = [&](double t) {
        double serviceTime;
        if (P.mode == "exp") {
            serviceTime = Exp(meanSvc);
        } else if (P.mode == "const") {
            serviceTime = meanSvc;
        } else {
            serviceTime = (sizeState ? P.s2 : P.s1) * 8.0 / P.C;
            const double u = U(rng);
            if (u < pSwitch) sizeState = 1 - sizeState;
        }
        if (t >= warm) serviceAcf.add(serviceTime);
        tDeparture = t + serviceTime;
    };

    while (true) {
        const double t = std::min(tArrival, tDeparture);
        if (t > T) break;

        advanceTime(t);

        if (tArrival <= tDeparture) {
            // Arrival event.
            if (t >= warm) {
                if (previousArrival >= warm) interarrivalAcf.add(t - previousArrival);
                ++R.off;

                if (n == K) {
                    ++R.lost;
                    if (currentLossRun == 0) ++R.runs;
                    ++currentLossRun;
                    if (!recoveryActive) {
                        recoveryActive = true;
                        recoveryStart = t;
                    }
                } else {
                    if (currentLossRun > 0) finishLossRun(currentLossRun);
                    currentLossRun = 0;
                }
            }
            previousArrival = t;

            if (n < K) {
                ++n;
                fifo.push_back(Packet{t, t >= warm});
                if (n == K && !fullActive) {
                    fullActive = true;
                    fullStart = t;
                }
                if (n == 1) startService(t);
            }
            tArrival = nextArrival(t);
        } else {
            // Departure event.
            if (n == K && fullActive) {
                fullActive = false;
                if (t > warm) {
                    fullDurationSum += t - std::max(fullStart, warm);
                    ++R.fullN;
                }
            }

            --n;
            tDeparture = std::numeric_limits<double>::infinity();

            const Packet packet = fifo.front();
            fifo.pop_front();

            if (packet.inMeasurementCohort) {
                const double delay = t - packet.arrival;
                delaySum += delay;
                ++delaySeen;

                if (delayReservoir.size() < RESERVOIR_CAPACITY) {
                    delayReservoir.push_back(delay);
                } else {
                    const unsigned long long j =
                        static_cast<unsigned long long>(rngDelay()) % delaySeen;
                    if (j < RESERVOIR_CAPACITY)
                        delayReservoir[static_cast<size_t>(j)] = delay;
                }
            }

            if (n > 0) startService(t);

            if (recoveryActive && n <= R.threshold) {
                recoveryDurations.push_back(t - recoveryStart);
                ++R.recN;
                recoveryActive = false;
            }
        }
    }

    advanceTime(T);

    // A loss run still open at T is an observed, right-truncated run.
    if (currentLossRun > 0) {
        finishLossRun(currentLossRun);
        R.lossRunCensored = 1;
    }

    if (recoveryActive) {
        R.recCens = 1;
        R.recCensDur = T - recoveryStart;
    }

    // A buffer-full period still active at T is right-censored.
    if (fullActive) {
        R.fullCens = 1;
        R.fullCensoredDuration = T - std::max(fullStart, warm);
    }

    // Count measurement-cohort packets still in the system at T as right-censored delays.
    for (const Packet& packet : fifo) {
        if (packet.inMeasurementCohort) ++R.delayCensored;
    }

    const double measurementTime = T - warm;
    R.p = R.off > 0 ? static_cast<double>(R.lost) / R.off : 0.0;
    R.meanRun = R.runs > 0 ? static_cast<double>(R.lost) / R.runs : 0.0;
    R.burst = R.meanRun * (1.0 - R.p);
    R.meanQ = areaQ / measurementTime;
    R.fullFrac = areaFull / measurementTime;

    R.fullMeanDur = R.fullN > 0
        ? fullDurationSum / static_cast<double>(R.fullN)
        : std::numeric_limits<double>::quiet_NaN();
    R.fullCensDur = R.fullCens ? R.fullCensoredDuration
                               : std::numeric_limits<double>::quiet_NaN();

    if (!recoveryDurations.empty()) {
        double sum = 0.0;
        for (double x : recoveryDurations) sum += x;
        R.recMean = sum / static_cast<double>(recoveryDurations.size());
        R.recMed = percentile(recoveryDurations, 0.50);
        R.recP95 = percentile(recoveryDurations, 0.95);
        R.recMax = *std::max_element(recoveryDurations.begin(), recoveryDurations.end());
    }

    if (delaySeen > 0) {
        R.delayCompleted = static_cast<long long>(delaySeen);
        R.delayMean = delaySum / static_cast<double>(delaySeen);
        R.delayP95 = percentile(delayReservoir, 0.95);

        // This is a finite-window diagnostic, not a formal steady-state validation.
        const double acceptedRate = static_cast<double>(R.off - R.lost) / measurementTime;
        if (acceptedRate > 0.0 && R.delayMean > 0.0)
            R.little = R.meanQ / (acceptedRate * R.delayMean);
    }

    R.acfInter = interarrivalAcf.value();
    R.acfSvc = serviceAcf.value();
    return R;
}

static double tCritical95(int df) {
    static const double table[] = {
        0.0, 12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262,
        2.228, 2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093,
        2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045,
        2.042
    };
    return (df >= 1 && df <= 30) ? table[df] : 1.96;
}

static void printOne(const Params& P, const Result& R, unsigned long long seed) {
    std::printf("mode=%s rho=%.3f K=%d h=%.3f phi=%.3f\n",
                P.mode.c_str(), P.rho, P.K, P.h, P.phi);
    std::printf("offered=%lld lost=%lld lossRatio=%.8f burstRatio=%.5f "
                "meanQ=%.5f fullFrac=%.5f\n",
                R.off, R.lost, R.p, R.burst, R.meanQ, R.fullFrac);
    std::printf("seed=%llu\n", seed);
    std::printf("lossRuns=%lld meanRunLen=%.5f maxRunLen=%lld | "
                "runs>=2: %.2f%% of runs, %.2f%% of lost packets; finalRunRightCensored=%d\n",
                R.runs, R.meanRun, R.maxRun,
                R.runs ? 100.0 * R.runsGE2 / R.runs : 0.0,
                R.lost ? 100.0 * R.lostInGE2 / R.lost : 0.0, R.lossRunCensored);
    std::printf("runLenDist(%% of observed runs):");
    for (int i = 1; i <= P.maxHist; ++i) {
        std::printf(" %d%s:%.2f", i, i == P.maxHist ? "+" : "",
                    R.runs ? 100.0 * R.hist[static_cast<size_t>(i)] / R.runs : 0.0);
    }
    std::printf("\n");
    std::printf("recovery(theta=%.2f, threshold=%d packets): completed=%lld "
                "censored=%d censoredDuration=%.6f s | mean=%.6f median=%.6f "
                "p95=%.6f max=%.6f s\n",
                P.theta, R.threshold, R.recN, R.recCens, R.recCensDur,
                R.recMean, R.recMed, R.recP95, R.recMax);
    std::printf("overflow periods: completed=%lld meanDuration=%.6f s; "
                "right-censored=%d censoredDuration=%.6f s\n",
                R.fullN, R.fullMeanDur, R.fullCens, R.fullCensDur);
    std::printf("delay cohort (accepted arrivals in [warm,T]): completed=%lld "
                "right-censored=%lld | mean=%.8f s p95=%.8f s\n",
                R.delayCompleted, R.delayCensored, R.delayMean, R.delayP95);
    std::printf("Little diagnostic meanQ/(acceptedRate*completedMeanDelay)=%.5f\n",
                R.little);
    std::printf("measured lag-1 autocorrelation: interarrival=%.5f service=%.5f\n",
                R.acfInter, R.acfSvc);
}

int main(int argc, char** argv) {
    try {
        const std::set<std::string> known = {
            "rho", "K", "h", "soj", "phi", "s1", "s2", "C", "simTime",
            "warm", "seed", "mode", "theta", "reps", "csv", "verbose", "maxHist"
        };
        std::map<std::string, std::string> args;

        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            const size_t equal = arg.find('=');
            if (arg.rfind("--", 0) != 0 || equal == std::string::npos || equal == 2)
                throw std::runtime_error("bad argument '" + arg + "' (use --name=value)");
            const std::string key = arg.substr(2, equal - 2);
            if (!known.count(key)) throw std::runtime_error("unknown option --" + key);
            if (args.count(key)) throw std::runtime_error("duplicate option --" + key);
            args[key] = arg.substr(equal + 1);
        }

        auto getDouble = [&](const std::string& key, double fallback) {
            return args.count(key) ? parseDouble(args.at(key), key) : fallback;
        };
        auto getInt = [&](const std::string& key, long long fallback) {
            return args.count(key) ? parseInteger(args.at(key), key) : fallback;
        };

        Params P;
        P.rho = getDouble("rho", P.rho);
        P.h = getDouble("h", P.h);
        P.soj = getDouble("soj", P.soj);
        P.phi = getDouble("phi", P.phi);
        P.s1 = getDouble("s1", P.s1);
        P.s2 = getDouble("s2", P.s2);
        P.C = getDouble("C", 10.0) * 1e6; // CLI C is in Mbps.
        P.T = getDouble("simTime", P.T);
        P.warm = getDouble("warm", P.warm);
        P.theta = getDouble("theta", P.theta);
        const long long kValue = getInt("K", P.K);
        const long long histValue = getInt("maxHist", P.maxHist);
        const long long repsValue = getInt("reps", 1);
        const long long seedValue = args.count("seed")
            ? static_cast<long long>(parseSeed(args.at("seed")))
            : static_cast<long long>(P.seed);
        if (kValue > std::numeric_limits<int>::max() ||
            histValue > std::numeric_limits<int>::max() ||
            repsValue > std::numeric_limits<int>::max())
            throw std::runtime_error("K, maxHist, or reps is too large");
        if (seedValue < 0) throw std::runtime_error("seed is too large for this build");
        P.K = static_cast<int>(kValue);
        P.maxHist = static_cast<int>(histValue);
        P.seed = static_cast<unsigned long long>(seedValue);
        P.mode = args.count("mode") ? args.at("mode") : P.mode;
        const int reps = static_cast<int>(repsValue);
        const bool verbose = getDouble("verbose", 0.0) != 0.0;

        if (P.mode != "exp" && P.mode != "const" && P.mode != "corr")
            throw std::runtime_error("mode must be exp|const|corr");
        if (!(P.rho > 0.0)) throw std::runtime_error("rho must be > 0");
        if (P.K < 1) throw std::runtime_error("K must be >= 1");
        if (!(P.h >= 1.0)) throw std::runtime_error("h must be >= 1 (h=1 means Poisson)");
        if (!(P.soj > 0.0)) throw std::runtime_error("soj must be > 0");
        if (!(P.phi >= -1.0 && P.phi <= 1.0))
            throw std::runtime_error("phi must be in [-1,1]");
        if (!(P.s1 > 0.0 && P.s2 > 0.0))
            throw std::runtime_error("s1 and s2 must be > 0");
        if (!(P.C > 0.0)) throw std::runtime_error("C must be > 0");
        if (!(P.T > 0.0 && P.warm >= 0.0 && P.warm < P.T))
            throw std::runtime_error("need 0 <= warm < simTime");
        if (!(P.theta > 0.0 && P.theta <= 1.0))
            throw std::runtime_error("theta must be in (0,1]");
        if (reps < 1) throw std::runtime_error("reps must be >= 1");
        if (P.maxHist < 2 || P.maxHist > 100)
            throw std::runtime_error("maxHist must be in [2,100]");
        if (P.mode != "corr" && args.count("phi") && P.phi != 0.0)
            std::fprintf(stderr, "WARNING: --phi is ignored unless --mode=corr\n");
        if (P.mode != "corr" && (args.count("s1") || args.count("s2")))
            std::fprintf(stderr, "NOTE: mode=%s uses only the mean packet size\n", P.mode.c_str());
        if (P.mode == "corr" && P.phi == 1.0)
            std::fprintf(stderr, "WARNING: phi=1 freezes the size state; individual runs may not "
                                 "have a 50/50 size mix. Use phi<1 for a mixing chain.\n");

        const double rhoLow = 2.0 * P.rho / (1.0 + P.h);
        const double rhoHigh = 2.0 * P.rho * P.h / (1.0 + P.h);
        std::fprintf(stderr, "info: per-state offered load rhoLow=%.4f rhoHigh=%.4f%s\n",
                     rhoLow, rhoHigh,
                     rhoHigh > 1.0 ? " (high-rate state is overloaded)" : "");
        if (P.T - P.warm < 50.0 * P.soj)
            std::fprintf(stderr, "WARNING: measurement window is short relative to MMPP "
                                 "mean sojourn time\n");

        std::ofstream csv;
        if (args.count("csv")) {
            const std::string csvName = args.at("csv");
            std::ifstream existing(csvName, std::ios::binary);
            if (existing.good())
                throw std::runtime_error("CSV file already exists; choose a new filename: " + csvName);
            csv.open(csvName, std::ios::out | std::ios::trunc);
            if (!csv) throw std::runtime_error("cannot create CSV file: " + csvName);

            csv << "mode,rho,K,h,soj,phi,theta,seed,offered,lost,lossRatio,burstRatio,"
                   "lossRuns,meanRunLen,maxRunLen,pctRunsGE2,pctLostInRunsGE2,lossRunRightCensored";
            for (int i = 1; i <= P.maxHist; ++i)
                csv << ",runLen" << i << (i == P.maxHist ? "plus" : "");
            csv << ",meanQ,fullFrac,overflowCompleted,overflowMeanDur,overflowCensored,"
                   "overflowCensoredDur,recoveryCompleted,recoveryCensored,recoveryCensoredDur,"
                   "recoveryMean,recoveryMedian,recoveryP95,recoveryMax,delayCompleted,"
                   "delayCensored,delayMean,delayP95,littleDiagnostic,acfInter,acfService\n";
            csv << std::setprecision(12);
        }

        struct Metric {
            std::string name;
            std::vector<double> values;
        };
        std::vector<Metric> metrics = {
            {"lossRatio", {}}, {"burstRatio", {}}, {"meanRunLen", {}},
            {"maxRunLen", {}}, {"pctRunsGE2", {}}, {"pctLostInRunsGE2", {}},
            {"meanQ", {}}, {"fullFrac", {}}, {"overflowMeanDur", {}},
            {"overflowCensored", {}}, {"recoveryMean", {}}, {"recoveryMedian", {}},
            {"recoveryP95", {}}, {"recoveryCensored", {}}, {"delayMean", {}},
            {"delayP95", {}}, {"delayCensored", {}}, {"littleDiagnostic", {}},
            {"acfInter", {}}, {"acfService", {}}
        };

        for (int r = 0; r < reps; ++r) {
            if (P.seed > std::numeric_limits<unsigned long long>::max() -
                             static_cast<unsigned long long>(r))
                throw std::runtime_error("seed + replication index overflows");
            const unsigned long long seed = P.seed + static_cast<unsigned long long>(r);
            const Result R = simulate(P, seed);

            if (reps == 1 || verbose) printOne(P, R, seed);

            const double pctRuns = R.runs ? 100.0 * R.runsGE2 / R.runs : 0.0;
            const double pctLost = R.lost ? 100.0 * R.lostInGE2 / R.lost : 0.0;
            const double values[] = {
                R.p, R.burst, R.meanRun, static_cast<double>(R.maxRun),
                pctRuns, pctLost, R.meanQ, R.fullFrac, R.fullMeanDur,
                static_cast<double>(R.fullCens), R.recMean, R.recMed, R.recP95,
                static_cast<double>(R.recCens), R.delayMean, R.delayP95,
                static_cast<double>(R.delayCensored), R.little, R.acfInter, R.acfSvc
            };
            for (size_t i = 0; i < metrics.size(); ++i)
                metrics[i].values.push_back(values[i]);

            if (csv.is_open()) {
                csv << P.mode << ',' << P.rho << ',' << P.K << ',' << P.h << ','
                    << P.soj << ',' << P.phi << ',' << P.theta << ',' << seed << ','
                    << R.off << ',' << R.lost << ',' << R.p << ',' << R.burst << ','
                    << R.runs << ',' << R.meanRun << ',' << R.maxRun << ','
                    << pctRuns << ',' << pctLost << ',' << R.lossRunCensored;
                for (int i = 1; i <= P.maxHist; ++i)
                    csv << ',' << R.hist[static_cast<size_t>(i)];
                csv << ',' << R.meanQ << ',' << R.fullFrac << ',' << R.fullN << ','
                    << R.fullMeanDur << ',' << R.fullCens << ',' << R.fullCensDur << ','
                    << R.recN << ',' << R.recCens << ',' << R.recCensDur << ','
                    << R.recMean << ',' << R.recMed << ',' << R.recP95 << ',' << R.recMax
                    << ',' << R.delayCompleted << ',' << R.delayCensored << ','
                    << R.delayMean << ',' << R.delayP95 << ',' << R.little << ','
                    << R.acfInter << ',' << R.acfSvc << '\n';
            }
        }

        if (reps > 1) {
            std::printf("\nmode=%s rho=%.3f K=%d h=%.3f phi=%.3f theta=%.2f | "
                        "%d replications (seeds %llu..%llu), simTime=%.3f warm=%.3f\n",
                        P.mode.c_str(), P.rho, P.K, P.h, P.phi, P.theta, reps,
                        P.seed, P.seed + static_cast<unsigned long long>(reps - 1),
                        P.T, P.warm);
            std::printf("%-22s %14s %16s\n", "metric", "mean", "+/- 95% CI");
            for (const Metric& metric : metrics) {
                double sum = 0.0;
                int count = 0;
                for (double x : metric.values) {
                    if (std::isfinite(x)) {
                        sum += x;
                        ++count;
                    }
                }
                if (count == 0) {
                    std::printf("%-22s %14s %16s\n", metric.name.c_str(), "n/a", "n/a");
                    continue;
                }
                const double mean = sum / count;
                double sumSquares = 0.0;
                for (double x : metric.values)
                    if (std::isfinite(x)) sumSquares += (x - mean) * (x - mean);
                const double ci = count > 1
                    ? tCritical95(count - 1) *
                          std::sqrt(sumSquares / (count - 1) / count)
                    : std::numeric_limits<double>::quiet_NaN();
                std::printf("%-22s %14.7g %16.5g\n", metric.name.c_str(), mean, ci);
            }
        }
        if (csv.is_open())
            std::fprintf(stderr, "CSV written to: %s\n", args.at("csv").c_str());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 2;
    }
}
