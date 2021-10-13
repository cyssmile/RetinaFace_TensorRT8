#include <glog/logging.h>
#include <gtest/gtest.h>

int main(int argc, char** argv) {
	testing::InitGoogleTest(&argc, argv);
	//testing::AddGlobalTestEnvironment(new testing::TestEnvironment);
	int ret = RUN_ALL_TESTS();
	return ret;
}