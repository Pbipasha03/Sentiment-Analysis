import { Link, useLocation } from "wouter";
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarProvider,
} from "@/components/ui/sidebar";
import { BarChart3, Cloud, Columns, FileOutput, Home, Library, Settings, MessageSquare } from "lucide-react";
import React from "react";

const NAV_ITEMS = [
  { title: "Dashboard", url: "/", icon: Home },
  { title: "Analyze Text", url: "/analyze", icon: MessageSquare },
  { title: "Batch Analysis", url: "/batch", icon: Library },
  { title: "Model Performance", url: "/models", icon: BarChart3 },
  { title: "Compare Models", url: "/compare", icon: Columns },
  { title: "Word Cloud", url: "/wordcloud", icon: Cloud },
  { title: "Export Report", url: "/report", icon: FileOutput },
];

export function Layout({ children }: { children: React.ReactNode }) {
  const [location] = useLocation();

  return (
    <SidebarProvider>
      <div className="flex h-screen w-full bg-background overflow-hidden text-foreground">
        <Sidebar className="border-r">
          <SidebarContent>
            <div className="p-4 py-6">
              <h2 className="text-xl font-bold tracking-tight text-primary flex items-center gap-2">
                <BarChart3 className="w-5 h-5 text-primary" />
                Microtext
              </h2>
              <p className="text-xs text-muted-foreground mt-1">Sentiment Analysis Tool</p>
            </div>
            <SidebarGroup>
              <SidebarGroupLabel>Analysis & Research</SidebarGroupLabel>
              <SidebarGroupContent>
                <SidebarMenu>
                  {NAV_ITEMS.map((item) => (
                    <SidebarMenuItem key={item.title}>
                      <SidebarMenuButton
                        asChild
                        isActive={location === item.url}
                        tooltip={item.title}
                      >
                        <Link href={item.url} className="flex items-center gap-3">
                          <item.icon className="w-4 h-4" />
                          <span>{item.title}</span>
                        </Link>
                      </SidebarMenuButton>
                    </SidebarMenuItem>
                  ))}
                </SidebarMenu>
              </SidebarGroupContent>
            </SidebarGroup>
          </SidebarContent>
        </Sidebar>
        <main className="flex-1 overflow-y-auto p-8 relative">
          <div className="max-w-6xl mx-auto w-full">
            {children}
          </div>
        </main>
      </div>
    </SidebarProvider>
  );
}
