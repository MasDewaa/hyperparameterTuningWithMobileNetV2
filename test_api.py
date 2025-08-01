#!/usr/bin/env python3
"""
API Testing Script for Batik Classification Service
Author: AI Assistant
Date: 2024
"""

import requests
import json
import time
import argparse
from pathlib import Path
from PIL import Image
import numpy as np

class BatikAPITester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
    def test_health_check(self):
        """Test health check endpoint"""
        print("🏥 Testing health check...")
        
        try:
            response = self.session.get(f"{self.base_url}/")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Health check passed!")
                print(f"   Response: {data}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    def test_prediction_without_file(self):
        """Test prediction endpoint without file (should fail)"""
        print("\n📤 Testing prediction without file...")
        
        try:
            response = self.session.post(f"{self.base_url}/predict")
            
            if response.status_code in [422, 400]:  # Expected for missing file
                print("✅ Prediction endpoint accessible (correctly rejected missing file)")
                return True
            else:
                print(f"❌ Unexpected response: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Prediction test error: {e}")
            return False
    
    def create_test_image(self, size=(224, 224), color=(255, 0, 0)):
        """Create a test image for testing"""
        print(f"\n🎨 Creating test image ({size[0]}x{size[1]})...")
        
        # Create a simple test image
        image = Image.new('RGB', size, color)
        
        # Add some noise to make it more realistic
        img_array = np.array(image)
        noise = np.random.randint(0, 50, img_array.shape, dtype=np.uint8)
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        
        test_image = Image.fromarray(img_array)
        test_image.save("test_batik.jpg")
        
        print("✅ Test image created: test_batik.jpg")
        return "test_batik.jpg"
    
    def test_prediction_with_file(self, image_path="test_batik.jpg"):
        """Test prediction with actual image file"""
        print(f"\n📤 Testing prediction with file: {image_path}")
        
        if not Path(image_path).exists():
            print(f"❌ Image file not found: {image_path}")
            return False
        
        try:
            with open(image_path, 'rb') as f:
                files = {'file': f}
                response = self.session.post(f"{self.base_url}/predict", files=files)
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Prediction successful!")
                print(f"   Predicted class: {data['data']['class_name']}")
                print(f"   Confidence: {data['data']['confidence']:.4f}")
                print(f"   Top 3 predictions:")
                for i, (label, prob) in enumerate(list(data['data']['probabilities'].items())[:3]):
                    print(f"     {i+1}. {label}: {prob:.4f}")
                return True
            else:
                print(f"❌ Prediction failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return False
    
    def test_performance(self, num_requests=5):
        """Test API performance"""
        print(f"\n⚡ Testing performance ({num_requests} requests)...")
        
        if not Path("test_batik.jpg").exists():
            self.create_test_image()
        
        times = []
        successes = 0
        
        for i in range(num_requests):
            start_time = time.time()
            
            try:
                with open("test_batik.jpg", 'rb') as f:
                    files = {'file': f}
                    response = self.session.post(f"{self.base_url}/predict", files=files)
                
                if response.status_code == 200:
                    end_time = time.time()
                    request_time = end_time - start_time
                    times.append(request_time)
                    successes += 1
                    print(f"   Request {i+1}: {request_time:.3f}s")
                else:
                    print(f"   Request {i+1}: Failed ({response.status_code})")
                    
            except Exception as e:
                print(f"   Request {i+1}: Error - {e}")
        
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            print(f"\n📊 Performance Results:")
            print(f"   Successful requests: {successes}/{num_requests}")
            print(f"   Average response time: {avg_time:.3f}s")
            print(f"   Min response time: {min_time:.3f}s")
            print(f"   Max response time: {max_time:.3f}s")
            
            if avg_time < 1.0:
                print("   ✅ Performance: Excellent (< 1s)")
            elif avg_time < 2.0:
                print("   ✅ Performance: Good (< 2s)")
            elif avg_time < 5.0:
                print("   ⚠️  Performance: Acceptable (< 5s)")
            else:
                print("   ❌ Performance: Poor (> 5s)")
            
            return True
        else:
            print("❌ No successful requests for performance testing")
            return False
    
    def test_error_handling(self):
        """Test error handling"""
        print("\n🚨 Testing error handling...")
        
        # Test with invalid file type
        print("   Testing invalid file type...")
        try:
            with open("test_api.py", 'rb') as f:  # Send Python file instead of image
                files = {'file': f}
                response = self.session.post(f"{self.base_url}/predict", files=files)
            
            if response.status_code in [400, 422]:
                print("   ✅ Correctly rejected invalid file type")
            else:
                print(f"   ❌ Unexpected response for invalid file: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error testing invalid file: {e}")
        
        # Test with very large file
        print("   Testing large file...")
        try:
            # Create a large dummy file
            with open("large_test.jpg", 'wb') as f:
                f.write(b'0' * 20 * 1024 * 1024)  # 20MB file
            
            with open("large_test.jpg", 'rb') as f:
                files = {'file': f}
                response = self.session.post(f"{self.base_url}/predict", files=files)
            
            if response.status_code in [400, 413]:
                print("   ✅ Correctly rejected large file")
            else:
                print(f"   ❌ Unexpected response for large file: {response.status_code}")
            
            # Clean up
            Path("large_test.jpg").unlink(missing_ok=True)
                
        except Exception as e:
            print(f"   ❌ Error testing large file: {e}")
        
        return True
    
    def test_cors(self):
        """Test CORS headers"""
        print("\n🌐 Testing CORS headers...")
        
        try:
            response = self.session.options(f"{self.base_url}/predict")
            
            cors_headers = [
                'Access-Control-Allow-Origin',
                'Access-Control-Allow-Methods',
                'Access-Control-Allow-Headers'
            ]
            
            found_headers = []
            for header in cors_headers:
                if header in response.headers:
                    found_headers.append(header)
            
            if found_headers:
                print(f"   ✅ CORS headers found: {', '.join(found_headers)}")
                return True
            else:
                print("   ⚠️  No CORS headers found")
                return False
                
        except Exception as e:
            print(f"   ❌ CORS test error: {e}")
            return False
    
    def run_comprehensive_test(self):
        """Run all tests"""
        print("🧪 Comprehensive API Testing")
        print("=" * 40)
        
        tests = [
            ("Health Check", self.test_health_check),
            ("Prediction without file", self.test_prediction_without_file),
            ("Prediction with file", lambda: self.test_prediction_with_file()),
            ("Performance", lambda: self.test_performance()),
            ("Error handling", self.test_error_handling),
            ("CORS headers", self.test_cors)
        ]
        
        results = []
        for test_name, test_func in tests:
            try:
                result = test_func()
                results.append((test_name, result))
            except Exception as e:
                print(f"❌ {test_name} failed with exception: {e}")
                results.append((test_name, False))
        
        # Summary
        print("\n📋 Test Summary:")
        print("=" * 40)
        
        passed = 0
        total = len(results)
        
        for test_name, result in results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   {test_name}: {status}")
            if result:
                passed += 1
        
        print(f"\n🎯 Overall Result: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All tests passed! API is working correctly.")
        elif passed >= total * 0.8:
            print("✅ Most tests passed. API is working well.")
        else:
            print("⚠️  Several tests failed. Please check the API implementation.")
        
        # Cleanup
        Path("test_batik.jpg").unlink(missing_ok=True)
        
        return passed == total

def main():
    parser = argparse.ArgumentParser(description="Test Batik Classification API")
    parser.add_argument("--url", default="http://localhost:8000", 
                       help="API base URL")
    parser.add_argument("--image", help="Path to test image")
    parser.add_argument("--performance", type=int, default=5,
                       help="Number of performance test requests")
    parser.add_argument("--comprehensive", action="store_true",
                       help="Run comprehensive test suite")
    
    args = parser.parse_args()
    
    tester = BatikAPITester(args.url)
    
    if args.comprehensive:
        success = tester.run_comprehensive_test()
        exit(0 if success else 1)
    
    # Individual tests
    if args.image:
        tester.test_prediction_with_file(args.image)
    else:
        tester.test_health_check()
        tester.test_prediction_without_file()
        tester.create_test_image()
        tester.test_prediction_with_file()
        tester.test_performance(args.performance)

if __name__ == "__main__":
    main() 