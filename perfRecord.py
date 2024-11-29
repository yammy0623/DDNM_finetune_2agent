import time

class PerformanceRecord():
    def __init__(self) -> None:
        self.stepsPerEpoch = []
        self.stepCount = 0
        self.stepStart = 0
        self.stepTime = 0
        self.ddimStart = 0
        self.ddimStepTime = 0
        self.rewardStart = 0
        self.rewardTime = 0
        self.resetStart = 0
        self.resetTime = 0
        self.resetCount = 0
        self.stepTimeRecorded = False
        self.ddimTimeRecorded = False
        self.rewardTimeRecorded = False
        self.resetTimeRecorded = False
    
    def stepTick(self):
        self.stepStart = time.time_ns()
    
    def stepTock(self):
        stepEnd = time.time_ns()
        if self.stepTime == 0:
            self.stepTime = stepEnd - self.stepStart
        else:
            self.stepTime = (self.stepTime + stepEnd - self.stepStart) / 2
        self.stepTimeRecorded = True

    def ddimTick(self):
        self.ddimStart = time.time_ns()
    
    def ddimTock(self):
        ddimEnd = time.time_ns()
        if self.ddimStepTime == 0:
            self.ddimStepTime = ddimEnd - self.ddimStart
        else:
            self.ddimStepTime = (self.ddimStepTime + ddimEnd - self.ddimStart) / 2
        self.ddimTimeRecorded = True

    def rewardTick(self):
        self.rewardStart = time.time_ns()
    
    def rewardTock(self):
        rewardEnd = time.time_ns()
        if self.rewardTime == 0:
            self.rewardTime = rewardEnd - self.rewardStart
        else:
            self.rewardTime = (self.rewardTime + rewardEnd - self.rewardStart) / 2
        self.rewardTimeRecorded = True

    def resetTick(self):
        self.resetStart = time.time_ns()
    
    def resetTock(self):
        resetEnd = time.time_ns()
        if self.resetTime == 0:
            self.resetTime = resetEnd - self.resetStart
        else:
            self.resetTime = (self.resetTime + resetEnd - self.resetStart) / 2
        self.resetTimeRecorded = True
        self.resetCount += 1

    def isStepTimeRecorded(self) -> bool:
        return self.stepTimeRecorded
    
    def isDDIMTimeRecorded(self) -> bool:
        return self.ddimTimeRecorded

    def isResetTimeRecorded(self) -> bool:
        return self.resetTimeRecorded
    
    def isRewardTimeRecorded(self) -> bool:
        return self.rewardTimeRecorded
        
    def epochTick(self):
        self.stepCount += 1
    
    def epochTock(self):
        self.stepsPerEpoch.append(self.stepCount)
        self.stepCount = 0

    def printResults(self):
        print("\nPerformance results")
        if self.stepsPerEpoch:
            average_steps = sum(self.stepsPerEpoch) / len(self.stepsPerEpoch)
            print(f"Average steps per epoch: {average_steps}")
        else:
            print("No epochs recorded.")
        if self.stepTimeRecorded:
            stepTime_ms = self.stepTime / 1000
            print(f"Average time used per step: {stepTime_ms} microseconds")
        else:
            print("Step time not recorded.")
        if self.ddimTimeRecorded:
            ddimTime_ms = self.ddimStepTime / 1000
            print(f"Average time used per DDIM step: {ddimTime_ms} microseconds")
        else:
            print("DDIM time not recorded.")
        if self.rewardTimeRecorded:
            rewardTime_ms = self.rewardTime / 1000
            print(f"Average time used per reward: {rewardTime_ms} microseconds")
        else:
            print("Reward time not recorded.")
        if self.resetTimeRecorded:
            resetTime_ms = self.resetTime / 1000
            print(f"Average time used per reset: {resetTime_ms} microseconds")
            print(f"Reset was called {self.resetCount} times.")
        else:
            print("Reset time not recorded.")


global recorder
recorder = PerformanceRecord()
