"use client";
import React from "react";
import { useRouter } from "next/navigation";
import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import Link from "next/dist/client/link";

const Login = () => {
  const router = useRouter();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError("");
    setSuccess("");

    try {
      // Create form data for OAuth2PasswordRequestForm
      const formData = new URLSearchParams();
      formData.append('username', username);
      formData.append('password', password);

      const response = await fetch("http://localhost:8000/token", {
        method: "POST",
        headers: {
          "Content-Type": "application/x-www-form-urlencoded",
        },
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        setError(data.detail || "Login failed. Please try again.");
        return;
      }

      setSuccess("Login successful!");
      localStorage.setItem("token", data.access_token); // Store token in local storage
      router.push("/"); // Redirect to dashboard or home page
    } catch (error) {
      setError("An unexpected error occurred. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100">
      <Card className="w-full max-w-md p-6">
        <CardHeader>
          <CardTitle>Login</CardTitle>
          <CardDescription>Please enter your credentials</CardDescription>
        </CardHeader>
        <CardContent>
          <form className="space-y-4" onSubmit={handleSubmit}>
            <Label htmlFor="username">Username</Label>
            <Input
              id="username"
              type="text"
              required
              value={username}
              onChange={(e) => setUsername(e.target.value)}
            />
            <Label htmlFor="password">Password</Label>
            <Input
              id="password"
              type="password"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
            <div className="flex items-center space-x-4 mt-4">
              <Button type="submit" disabled={isLoading}>
                {isLoading ? "Signing in..." : "Sign in"}
              </Button>
              <Link className="text-black hover:underline" href="/signup">
                not registered?
              </Link>
            </div>
          </form>
        </CardContent>
        {error && <p className="text-red-500 mt-2">{error}</p>}
        {success && <p className="text-green-500 mt-2">{success}</p>}
      </Card>
    </div>
  );
};

export default Login; 