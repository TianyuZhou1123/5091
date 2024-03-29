﻿using System;

namespace HW_6
{
    public class Basicvalues
    {
        public static double[] Getvalues(double S0, double K, double r, double dividend, double sigma, double T, int N)
        {
            //sigma means volatility
            double dt = T / N;
            double dx = sigma * Math.Sqrt(3 * dt);
            double v = r - dividend - 0.5 * sigma * sigma;
            double edx = Math.Exp(dx);
            double pu = 0.5 * ((sigma * sigma * dt + v * v * dt * dt) / (dx * dx) + v * dt / dx);
            double pm = 1 - (sigma * sigma * dt + v * v * dt * dt) / (dx * dx);
            double pd = 0.5 * ((sigma * sigma * dt + v * v * dt * dt) / (dx * dx) - v * dt / dx);
            double disc = Math.Exp(-r * dt);
            double[] values = { dt, dx, v, edx, pu, pm, pd, disc };
            return values;
        }
    }
    public class Node_underlyingprice
    {
        public static double[,] Underlyingprice(double S0, double K, double r, double dividend, double sigma, double T, int N)
        {
            double dt = Basicvalues.Getvalues(S0, K, r, dividend, sigma, T, N)[0]; ;
            double dx = Basicvalues.Getvalues(S0, K, r, dividend, sigma, T, N)[1];
            double v = Basicvalues.Getvalues(S0, K, r, dividend, sigma, T, N)[2];
            double edx = Basicvalues.Getvalues(S0, K, r, dividend, sigma, T, N)[3];
            double[,] price = new double[2 * N + 1, N + 1];// rectangal of 2N+1 rows and N+1 columns
            price[N, 0] = S0;//the initial price of the trinomial tree
            for (int j = 1; j <= N; j++)
            {
                for (int i = N - j; i <= N + j; i++)
                {
                    price[i, j] = S0 * Math.Pow(edx, N - i);
                }
            }
            return price;
        }
    }
    public class Node_intrinsicvalues
    {
        public static double[,] Intrinsicvalues(double S0, double K, double r, double dividend, double sigma, double T, int N, string type)
        {
            double pu = Basicvalues.Getvalues(S0, K, r, dividend, sigma, T, N)[4];
            double pm = Basicvalues.Getvalues(S0, K, r, dividend, sigma, T, N)[5];
            double pd = Basicvalues.Getvalues(S0, K, r, dividend, sigma, T, N)[6];
            double disc = Basicvalues.Getvalues(S0, K, r, dividend, sigma, T, N)[7];
            double[,] underlyingprice = Node_underlyingprice.Underlyingprice(S0, K, r, dividend, sigma, T, N);
            double[,] price = new double[2 * N + 1, N + 1];
            for (int i = 0; i <underlyingprice.GetLength(0); i++)
            {
                int j = underlyingprice.GetLength(1) - 1;
                    if (type == "call")
                    {
                        price[i, j] = Math.Max(underlyingprice[i, j] - K, 0);
                    }
                    else
                    {
                        price[i, j] = Math.Max(K - underlyingprice[i, j], 0);
                    }
                
            }
            for (int j = N - 1; j >=0; j--)
            {
                for (int i = N - j; i <= N + j; i++)
                {
                    price[i, j] = disc * (pu * price[i - 1, j + 1] + pm * price[i , j+1] + pd * price[i + 1, j + 1]);
                }
            }
            return price;
        }
    }
    public class American_option
    {
        public static double[,] American(double S0, double K, double r, double dividend, double sigma, double T, int N, string type)
        {
            double[,] intrinsic = Node_intrinsicvalues.Intrinsicvalues(S0, K, r, dividend, sigma, T, N, type);
            double[,] Amer = new double[2 * N + 1, N + 1];
            for (int i = 0; i < 2*N+1; i++)
            {
                int j = N;
                Amer[i, j] = intrinsic[i, j];
                
            }
            double pu = Basicvalues.Getvalues(S0, K, r, dividend, sigma, T, N)[4];
            double pm = Basicvalues.Getvalues(S0, K, r, dividend, sigma, T, N)[5];
            double pd = Basicvalues.Getvalues(S0, K, r, dividend, sigma, T, N)[6];
            double disc = Basicvalues.Getvalues(S0, K, r, dividend, sigma, T, N)[7];
            double[,] underlyingprice = Node_underlyingprice.Underlyingprice(S0, K, r, dividend, sigma, T, N);
            for (int j = N -1; j >= 0; j--)
            {
                for (int i = N - j; i <= N + j; i++)
                {
                    Amer[i, j] = disc * (pu * Amer[i - 1, j + 1] + pm * Amer[i, j+1] + pd * Amer[i + 1, j + 1]);
                    if (type == "call")
                    {
                        Amer[i, j] = Math.Max(underlyingprice[i, j] - K, Amer[i, j]);//this is the difference  between Europ and America
                    }
                    else
                    {
                        Amer[i, j] = Math.Max(K - underlyingprice[i, j], Amer[i, j]);
                    }
                }
            }
            return Amer;
        }
    }
    public class Option_price
    {
        public static double[,] Optionprice(double S0, double K, double r, double dividend, double sigma, double T, int N, string type, string optiontype)
        {
            double[,] intrinsic = Node_intrinsicvalues.Intrinsicvalues(S0, K, r, dividend, sigma, T, N, type);
            double[,] optionprice = new double[2*N + 1, N + 1];
            double[,] american = American_option.American(S0, K, r, dividend, sigma, T, N, type);
            if (optiontype == "Europ")
            {
                optionprice = intrinsic;
            }
            else
            {
                optionprice = american;
            }
            return optionprice;
        }
    }
    public class Greek_values
    {
        public static double[] Greekvalues(double S0, double K, double r, double dividend, double sigma, double T, int N, string type, string optiontype)
        {
            double[,] underlyingprice = Node_underlyingprice.Underlyingprice(S0, K, r, dividend, sigma, T, N);
            double[,] optionprice = Option_price.Optionprice(S0, K, r, dividend, sigma, T, N, type, optiontype);
            int a = N;//use the first step
            int b = 1;
            double delta = (optionprice[a-1, b] - optionprice[a+1, b]) / (underlyingprice[a-1, b ] - underlyingprice[a+1, b ]);
            double gamma = ((optionprice[a-1, b ] - optionprice[a, b]) / (underlyingprice[a-1, b ] - underlyingprice[a, b]) - (optionprice[a, b] - optionprice[a+1, b ]) / (underlyingprice[a, b] - underlyingprice[a+1, b ])) / ((underlyingprice[a-1, b ] - underlyingprice[a+1, b ]) / 2);
            double[,] optionprice1 = Option_price.Optionprice(S0, K, r, dividend, sigma + 0.01 * sigma, T, N, type, optiontype);
            double[,] optionprice2 = Option_price.Optionprice(S0, K, r, dividend, sigma - 0.01 * sigma, T, N, type, optiontype);
            double vega = ((optionprice1[N,0] - optionprice2[N,0]) / (2 * 0.01 * sigma));
            double theta = (optionprice[a, b] - optionprice[a , b-1]) / (T / N);
            double[,] optionprice3 = Option_price.Optionprice(S0, K, r + 0.01 * r, dividend, sigma, T, N, type, optiontype);
            double[,] optionprice4 = Option_price.Optionprice(S0, K, r - 0.01 * r, dividend, sigma, T, N, type, optiontype);
            double rho = (optionprice3[N,0] - optionprice4[N,0]) / (2 * 0.01 * r);
            double[] values= { delta, gamma, vega, theta, rho };
            return values;

        }
    }
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Enter underlying price(S0): ");
            double S0 = Convert.ToDouble(Console.ReadLine());
            Console.WriteLine("Enter strike price(K): ");
            double K = Convert.ToDouble(Console.ReadLine());
            Console.WriteLine("Enter risk-free rate: ");
            double r = Convert.ToDouble(Console.ReadLine());
            Console.WriteLine("Enter dividend: ");
            double dividend = Convert.ToDouble(Console.ReadLine());
            Console.WriteLine("Enter volatility: ");
            double sigma = Convert.ToDouble(Console.ReadLine());
            Console.WriteLine("Enter tenor(T): ");
            double T = Convert.ToDouble(Console.ReadLine());
            Console.WriteLine("Enter numbers of steps(N): ");
            int N = Convert.ToInt32(Console.ReadLine());  
            Console.WriteLine("Enter type(call or put): ");
            string type = Convert.ToString(Console.ReadLine());
            Console.WriteLine("Enter optiontype(Europ or America): ");
            string optiontype = Convert.ToString(Console.ReadLine());
            double[,] option = Option_price.Optionprice(S0, K, r, dividend, sigma, T, N, type, optiontype);
            Console.WriteLine("Option price: "+option[N, 0]);//option price
            double[] greek = Greek_values.Greekvalues(S0, K, r, dividend, sigma, T, N,type,optiontype);
            Console.WriteLine("Delta: " + greek[0]);//delta
            Console.WriteLine("Gamma: " + greek[1]);//gamma
            Console.WriteLine("Vega: " + greek[2]);//vega
            Console.WriteLine("Theta: " + greek[3]);//theta
            Console.WriteLine("Rho: " + greek[4]);//rho
            Console.ReadLine();
        }

    }
}




